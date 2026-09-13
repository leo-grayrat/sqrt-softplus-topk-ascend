#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


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
                key = ((row.get("pipe") or "<unknown>").strip(),
                       (row.get("instr") or "<unknown>").strip())
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


def row(name: str, vals):
    return f"| {name} | {vals[0]:.0f} | {vals[1]:.0f} | {vals[2]:.3f} |"


def show_key(title: str, base, bulk, key):
    b = base.get(key, [0.0, 0.0, 0.0])
    n = bulk.get(key, [0.0, 0.0, 0.0])
    print(f"\n## {title}")
    print("| variant | calls | cycles | time_us |")
    print("| --- | ---: | ---: | ---: |")
    print(row("baseline", b))
    print(row("bulk", n))
    dc = n[0] - b[0]
    dy = n[1] - b[1]
    ratio = n[1] / b[1] if b[1] else float("inf")
    ratio_text = "inf" if ratio == float("inf") else f"{ratio:.3f}x"
    print(f"\nΔ calls={dc:+.0f}, Δ cycles={dy:+.0f}, cycle ratio={ratio_text}")


def top_delta(base, bulk, limit=15):
    rows = []
    for key in set(base) | set(bulk):
        b = base.get(key, [0.0, 0.0, 0.0])[1]
        n = bulk.get(key, [0.0, 0.0, 0.0])[1]
        rows.append((abs(n - b), n - b, key, b, n))
    print("\n## 最大 instruction cycle 差值")
    print("| pipe | instruction | baseline | bulk | delta |")
    print("| --- | --- | ---: | ---: | ---: |")
    for _, delta, (pipe, instr), b, n in sorted(rows, reverse=True)[:limit]:
        print(f"| {pipe} | `{instr}` | {b:.0f} | {n:.0f} | {delta:+.0f} |")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <profile-output root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    base_root = latest_opprof(root / "baseline")
    bulk_root = latest_opprof(root / "bulk")
    base = collect(base_root)
    bulk = collect(bulk_root)

    print(f"baseline OPPROF: {base_root}")
    print(f"bulk OPPROF: {bulk_root}")

    show_key("VECTOR BAR", base, bulk, ("VECTOR", "BAR"))
    show_key("VECTOR MOV_UB_TO_UB", base, bulk, ("VECTOR", "MOV_UB_TO_UB"))

    bp = pipes(base)
    np = pipes(bulk)
    print("\n## Pipe totals")
    print("| pipe | variant | calls | cycles | time_us |")
    print("| --- | --- | ---: | ---: | ---: |")
    for pipe in ("VECTOR", "SCALAR", "MTE2", "MTE3", "ALL"):
        print(f"| {pipe} | baseline | {bp[pipe][0]:.0f} | {bp[pipe][1]:.0f} | {bp[pipe][2]:.3f} |")
        print(f"| {pipe} | bulk | {np[pipe][0]:.0f} | {np[pipe][1]:.0f} | {np[pipe][2]:.3f} |")

    top_delta(base, bulk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
