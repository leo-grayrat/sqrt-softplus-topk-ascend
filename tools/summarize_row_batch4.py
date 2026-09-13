#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = ("bulk_control", "specialized_sequential", "batch4")


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


def reduction(a: float, b: float) -> float:
    return 1.0 - b / a if a else 0.0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <profile-output root>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    data = {}
    times = {}
    opprof = {}
    for variant in VARIANTS:
        opprof[variant] = latest_opprof(root / variant)
        data[variant] = collect(opprof[variant])
        times[variant] = parse_kernel_time(root / f"{variant}_msprof.log")
        print(f"{variant} OPPROF: {opprof[variant]}")

    print("\n## Kernel wall time")
    print("| variant | duration_us | running_time_us |")
    print("| --- | ---: | ---: |")
    for variant in VARIANTS:
        duration, running = times[variant]
        print(f"| {variant} | {duration:.3f} | {running:.3f} |")

    bulk = times["bulk_control"][1]
    spec = times["specialized_sequential"][1]
    batch = times["batch4"][1]
    print("\n## Pairwise attribution")
    print("| comparison | speedup | running-time reduction |")
    print("| --- | ---: | ---: |")
    print(f"| bulk_control -> specialized_sequential | {bulk/spec:.3f}x | {reduction(bulk, spec)*100:.1f}% |")
    print(f"| specialized_sequential -> batch4 | {spec/batch:.3f}x | {reduction(spec, batch)*100:.1f}% |")
    print(f"| bulk_control -> batch4 | {bulk/batch:.3f}x | {reduction(bulk, batch)*100:.1f}% |")

    keys = [
        ("VECTOR BAR", ("VECTOR", "BAR")),
        ("VECTOR MOV_UB_TO_UB", ("VECTOR", "MOV_UB_TO_UB")),
        ("VECTOR VBS32", ("VECTOR", "VBS32")),
        ("VECTOR VMS4V2", ("VECTOR", "VMS4V2")),
        ("SCALAR WAIT_FLAG", ("SCALAR", "WAIT_FLAG")),
        ("SCALAR ST_XD_XN_IMM", ("SCALAR", "ST_XD_XN_IMM")),
        ("SCALAR STI_XN_IMM", ("SCALAR", "STI_XN_IMM")),
        ("SCALAR STP_XI_XJ_XN", ("SCALAR", "STP_XI_XJ_XN")),
        ("SCALAR LD_XD_XN_IMM", ("SCALAR", "LD_XD_XN_IMM")),
    ]

    print("\n## Selected instructions")
    print("| instruction | variant | calls | cycles | instr_time_us |")
    print("| --- | --- | ---: | ---: | ---: |")
    for title, key in keys:
        for variant in VARIANTS:
            vals = data[variant].get(key, [0.0, 0.0, 0.0])
            print(f"| {title} | {variant} | {vals[0]:.0f} | {vals[1]:.0f} | {vals[2]:.3f} |")

    print("\n## Pipe totals")
    print("| pipe | variant | calls | cycles | instr_time_us |")
    print("| --- | --- | ---: | ---: | ---: |")
    totals = {variant: pipe_totals(data[variant]) for variant in VARIANTS}
    for pipe in ("SCALAR", "VECTOR", "MTE2", "MTE3", "ALL", "FLOWCTRL"):
        for variant in VARIANTS:
            vals = totals[variant][pipe]
            print(f"| {pipe} | {variant} | {vals[0]:.0f} | {vals[1]:.0f} | {vals[2]:.3f} |")

    print("\n## Attribution decision")
    spec_gain = reduction(bulk, spec)
    batch_gain = reduction(spec, batch)
    if batch_gain >= 0.20:
        print("BATCHING_POSITIVE: after controlling for specialization, cross-row batching still saves at least 20% running time.")
    elif batch_gain >= 0.10:
        print("BATCHING_MODEST: batching adds a real 10-20% gain beyond specialization, but is not a large-gap explanation.")
    elif batch_gain > 0.0:
        print("SPECIALIZATION_DOMINATES: most of the original 19% gain came from compile-time specialization; batching adds less than 10%.")
    else:
        print("NO_BATCHING_GAIN: after controlling for specialization, batch4 does not improve running time.")

    if spec_gain >= 0.10:
        print("SPECIALIZATION_SIGNAL: fixed-shape specialization alone saves at least 10%; consider production dispatch specialization separately.")
    else:
        print("SPECIALIZATION_SMALL: fixed-shape specialization alone saves less than 10%.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
