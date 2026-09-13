#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROWS = (1, 2, 4, 8)


def latest_opprof(root: Path) -> Path:
    matches = sorted(root.glob("OPPROF_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"{root} 下没有 OPPROF_*")
    return matches[-1]


def fnum(value: str | None) -> float:
    try:
        return float((value or "0").strip())
    except ValueError:
        return 0.0


def collect(root: Path):
    instr = defaultdict(lambda: [0.0, 0.0, 0.0])
    for path in root.rglob("*_instr_exe.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (
                    (row.get("pipe") or "<unknown>").strip(),
                    (row.get("instr") or "<unknown>").strip(),
                )
                instr[key][0] += fnum(row.get("call_count"))
                instr[key][1] += fnum(row.get("cycles"))
                instr[key][2] += fnum(row.get("running_time(us)"))
    if not instr:
        raise FileNotFoundError(f"{root} 下没有 instruction CSV 数据")
    return instr


def pipes(data):
    out = defaultdict(lambda: [0.0, 0.0, 0.0])
    for (pipe, _), vals in data.items():
        for i in range(3):
            out[pipe][i] += vals[i]
    return out


def fit_linear(xs, ys):
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return mean_y, 0.0, 1.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    intercept = mean_y - slope * mean_x
    pred = [intercept + slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return intercept, slope, r2


def show_pipe_table(data_by_rows):
    print("## Pipe totals")
    print("| rows | pipe | calls | cycles | time_us | cycles/row |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for rows in ROWS:
        totals = pipes(data_by_rows[rows])
        for pipe in ("SCALAR", "VECTOR", "MTE2", "MTE3", "ALL", "FLOWCTRL"):
            vals = totals[pipe]
            print(
                f"| {rows} | {pipe} | {vals[0]:.0f} | {vals[1]:.0f} | "
                f"{vals[2]:.3f} | {vals[1] / rows:.1f} |"
            )


def show_pipe_fits(data_by_rows):
    print("\n## Fixed + per-row cycle fit")
    print("| pipe | fixed_cycles | per_row_cycles | R2 | rows8/fixed |")
    print("| --- | ---: | ---: | ---: | ---: |")
    xs = list(ROWS)
    for pipe in ("SCALAR", "VECTOR", "MTE2", "MTE3", "FLOWCTRL"):
        ys = [pipes(data_by_rows[r])[pipe][1] for r in ROWS]
        fixed, per_row, r2 = fit_linear(xs, ys)
        ratio = ys[-1] / fixed if fixed > 0 else math.inf
        ratio_text = "inf" if math.isinf(ratio) else f"{ratio:.2f}"
        print(f"| {pipe} | {fixed:.1f} | {per_row:.1f} | {r2:.4f} | {ratio_text} |")


def top_scalar_keys(data_by_rows, limit=12):
    rows8 = data_by_rows[8]
    keys = [
        (key, vals[1])
        for key, vals in rows8.items()
        if key[0] == "SCALAR"
    ]
    keys.sort(key=lambda item: item[1], reverse=True)
    return [key for key, _ in keys[:limit]]


def show_scalar_hotspots(data_by_rows):
    keys = top_scalar_keys(data_by_rows)
    print("\n## SCALAR hotspots across rows")
    print("| instruction | r1 | r2 | r4 | r8 | fixed | per_row | R2 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    xs = list(ROWS)
    for key in keys:
        ys = [data_by_rows[r].get(key, [0.0, 0.0, 0.0])[1] for r in ROWS]
        fixed, per_row, r2 = fit_linear(xs, ys)
        print(
            f"| `{key[1]}` | {ys[0]:.0f} | {ys[1]:.0f} | {ys[2]:.0f} | {ys[3]:.0f} | "
            f"{fixed:.1f} | {per_row:.1f} | {r2:.4f} |"
        )


def show_selected(data_by_rows):
    names = ("ST_XD_XN_IMM", "STI_XN_IMM", "STP_XI_XJ_XN", "LD")
    print("\n## Selected scalar instructions")
    print("| instruction | r1 | r2 | r4 | r8 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name in names:
        vals = []
        for rows in ROWS:
            vals.append(data_by_rows[rows].get(("SCALAR", name), [0.0, 0.0, 0.0])[1])
        print(f"| `{name}` | {vals[0]:.0f} | {vals[1]:.0f} | {vals[2]:.0f} | {vals[3]:.0f} |")


def interpret(data_by_rows):
    scalar = [pipes(data_by_rows[r])["SCALAR"][1] for r in ROWS]
    fixed, per_row, r2 = fit_linear(list(ROWS), scalar)
    rows8 = scalar[-1]
    per_row_share = (per_row * 8.0 / rows8) if rows8 > 0 else 0.0

    print("\n## Interpretation")
    print(
        f"SCALAR fit: cycles ≈ {fixed:.1f} + {per_row:.1f} * rows, "
        f"R²={r2:.4f}; at rows=8, fitted repeated-row term is about {per_row_share * 100:.1f}% of SCALAR cycles."
    )
    if r2 >= 0.95 and per_row_share >= 0.65:
        print("Signal: SCALAR is strongly per-row dominated. Prioritize cross-row batching / reducing per-row control and tensor bookkeeping.")
    elif r2 >= 0.95 and per_row_share <= 0.35:
        print("Signal: SCALAR is largely fixed/setup dominated. Prioritize amortizing kernel/TPipe setup over more rows per block.")
    else:
        print("Signal: mixed or non-linear scaling. Inspect the hotspot table before choosing between setup amortization and per-row restructuring.")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <rows-scaling profile-output root>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    data_by_rows = {}
    for rows in ROWS:
        opprof = latest_opprof(root / f"rows_{rows}")
        data_by_rows[rows] = collect(opprof)
        print(f"rows={rows} OPPROF: {opprof}")

    print()
    show_pipe_table(data_by_rows)
    show_pipe_fits(data_by_rows)
    show_selected(data_by_rows)
    show_scalar_hotspots(data_by_rows)
    interpret(data_by_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
