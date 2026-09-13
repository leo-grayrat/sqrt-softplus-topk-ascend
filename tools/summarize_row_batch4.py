#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
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


def pipe_totals(data):
    out = defaultdict(lambda: [0.0, 0.0, 0.0])
    for (pipe, _), vals in data.items():
        for i in range(3):
            out[pipe][i] += vals[i]
    return out


def parse_kernel_time(log_path: Path) -> tuple[float, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"core0\.veccore0\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)",
        text,
    )
    if not matches:
        raise ValueError(f"无法从 {log_path} 解析 duration/running time")
    duration, running = matches[-1]
    return float(duration), float(running)


def show_key(title: str, control, batch, key):
    c = control.get(key, [0.0, 0.0, 0.0])
    b = batch.get(key, [0.0, 0.0, 0.0])
    ratio = b[1] / c[1] if c[1] else float("inf")
    ratio_text = "inf" if ratio == float("inf") else f"{ratio:.3f}x"
    print(f"\n## {title}")
    print("| variant | calls | cycles | instr_time_us |")
    print("| --- | ---: | ---: | ---: |")
    print(f"| bulk_control | {c[0]:.0f} | {c[1]:.0f} | {c[2]:.3f} |")
    print(f"| batch4 | {b[0]:.0f} | {b[1]:.0f} | {b[2]:.3f} |")
    print(f"\ncycle ratio={ratio_text}, delta={b[1]-c[1]:+.0f}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <profile-output root>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    control_root = latest_opprof(root / "bulk_control")
    batch_root = latest_opprof(root / "batch4")
    control = collect(control_root)
    batch = collect(batch_root)

    control_duration, control_running = parse_kernel_time(root / "bulk_control_msprof.log")
    batch_duration, batch_running = parse_kernel_time(root / "batch4_msprof.log")
    speedup = control_running / batch_running if batch_running else float("inf")
    reduction = 1.0 - batch_running / control_running if control_running else 0.0

    print(f"bulk_control OPPROF: {control_root}")
    print(f"batch4 OPPROF: {batch_root}")

    print("\n## Kernel wall time")
    print("| variant | duration_us | running_time_us |")
    print("| --- | ---: | ---: |")
    print(f"| bulk_control | {control_duration:.3f} | {control_running:.3f} |")
    print(f"| batch4 | {batch_duration:.3f} | {batch_running:.3f} |")
    print(f"\nrunning_time speedup={speedup:.3f}x, reduction={reduction*100:.1f}%")

    for title, key in [
        ("VECTOR BAR", ("VECTOR", "BAR")),
        ("VECTOR MOV_UB_TO_UB", ("VECTOR", "MOV_UB_TO_UB")),
        ("VECTOR VBS32", ("VECTOR", "VBS32")),
        ("VECTOR VMS4V2", ("VECTOR", "VMS4V2")),
        ("SCALAR WAIT_FLAG", ("SCALAR", "WAIT_FLAG")),
        ("SCALAR ST_XD_XN_IMM", ("SCALAR", "ST_XD_XN_IMM")),
        ("SCALAR STI_XN_IMM", ("SCALAR", "STI_XN_IMM")),
        ("SCALAR STP_XI_XJ_XN", ("SCALAR", "STP_XI_XJ_XN")),
    ]:
        show_key(title, control, batch, key)

    cp = pipe_totals(control)
    bp = pipe_totals(batch)
    print("\n## Pipe totals")
    print("| pipe | variant | calls | cycles | instr_time_us |")
    print("| --- | --- | ---: | ---: | ---: |")
    for pipe in ("SCALAR", "VECTOR", "MTE2", "MTE3", "ALL", "FLOWCTRL"):
        print(
            f"| {pipe} | bulk_control | {cp[pipe][0]:.0f} | {cp[pipe][1]:.0f} | {cp[pipe][2]:.3f} |"
        )
        print(
            f"| {pipe} | batch4 | {bp[pipe][0]:.0f} | {bp[pipe][1]:.0f} | {bp[pipe][2]:.3f} |"
        )

    deltas = []
    for key in set(control) | set(batch):
        c = control.get(key, [0.0, 0.0, 0.0])[1]
        b = batch.get(key, [0.0, 0.0, 0.0])[1]
        deltas.append((abs(b - c), b - c, key, c, b))

    print("\n## Largest instruction-cycle deltas")
    print("| pipe | instruction | bulk_control | batch4 | delta |")
    print("| --- | --- | ---: | ---: | ---: |")
    for _, delta, (pipe, instr), c, b in sorted(deltas, reverse=True)[:20]:
        print(f"| {pipe} | `{instr}` | {c:.0f} | {b:.0f} | {delta:+.0f} |")

    print("\n## Spike decision")
    ratio = batch_running / control_running if control_running else float("inf")
    if ratio <= 0.75:
        print("STRONG_POSITIVE: batch4 lowers simulator running_time by at least 25%.")
    elif ratio < 0.90:
        print("BORDERLINE: measurable batching gain, but below the predeclared strong threshold.")
    else:
        print("STOP: batch4 stays within 10% of bulk-control; do not rescue with B=8/bank/final-merge tuning.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
