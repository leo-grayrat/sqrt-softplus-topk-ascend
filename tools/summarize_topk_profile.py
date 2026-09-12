#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def to_float(value: str) -> float:
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return 0.0


def latest_opprof(root: Path) -> Path:
    matches = sorted(root.glob("OPPROF_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"{root} 下没有 OPPROF_* 目录")
    return matches[-1]


def collect_instr(root: Path):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    files = sorted(root.rglob("*_instr_exe.csv"))
    if not files:
        raise FileNotFoundError(f"{root} 下没有 *_instr_exe.csv")

    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                instr = (row.get("instr") or "<unknown>").strip()
                pipe = (row.get("pipe") or "<unknown>").strip()
                key = (pipe, instr)
                totals[key][0] += to_float(row.get("call_count", "0"))
                totals[key][1] += to_float(row.get("cycles", "0"))
                totals[key][2] += to_float(row.get("running_time(us)", "0"))
    return totals, files


def collect_code(root: Path):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    files = sorted(root.rglob("*_code_exe.csv"))
    if not files:
        return totals, files

    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                code = (row.get("code") or "<unknown>").strip()
                totals[code][0] += to_float(row.get("call_count", "0"))
                totals[code][1] += to_float(row.get("cycles", "0"))
                totals[code][2] += to_float(row.get("running_time(us)", "0"))
    return totals, files


def print_instr_table(title: str, totals, limit: int):
    print(f"\n## {title}")
    print("| pipe | instruction | calls | cycles | time_us |")
    print("| --- | --- | ---: | ---: | ---: |")
    rows = sorted(totals.items(), key=lambda item: item[1][1], reverse=True)[:limit]
    for (pipe, instr), (calls, cycles, time_us) in rows:
        safe_instr = instr.replace("|", "\\|")
        print(f"| {pipe} | `{safe_instr}` | {calls:.0f} | {cycles:.0f} | {time_us:.3f} |")


def print_code_table(title: str, totals, limit: int):
    print(f"\n## {title}")
    print("| code | calls | cycles | time_us |")
    print("| --- | ---: | ---: | ---: |")
    rows = sorted(totals.items(), key=lambda item: item[1][1], reverse=True)[:limit]
    for code, (calls, cycles, time_us) in rows:
        safe_code = code.replace("|", "\\|")
        print(f"| `{safe_code}` | {calls:.0f} | {cycles:.0f} | {time_us:.3f} |")


def print_delta(base, pivot, limit: int):
    print("\n## Pivot - baseline：instruction cycle 差值")
    print("| pipe | instruction | baseline_cycles | pivot_cycles | delta | ratio |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")

    keys = set(base) | set(pivot)
    rows = []
    for key in keys:
        b = base.get(key, [0.0, 0.0, 0.0])[1]
        p = pivot.get(key, [0.0, 0.0, 0.0])[1]
        delta = p - b
        ratio = p / b if b > 0 else float("inf") if p > 0 else 1.0
        rows.append((abs(delta), delta, ratio, key, b, p))

    for _, delta, ratio, (pipe, instr), b, p in sorted(rows, reverse=True)[:limit]:
        ratio_text = "inf" if ratio == float("inf") else f"{ratio:.2f}x"
        safe_instr = instr.replace("|", "\\|")
        print(f"| {pipe} | `{safe_instr}` | {b:.0f} | {p:.0f} | {delta:+.0f} | {ratio_text} |")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总 paired msprof simulator 的 baseline/pivot 指令与代码行开销")
    parser.add_argument(
        "profile_root",
        type=Path,
        help="包含 baseline/ 和 pivot/ 的 profile-output/... 目录",
    )
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    base_root = latest_opprof(args.profile_root / "baseline") / "simulator"
    pivot_root = latest_opprof(args.profile_root / "pivot") / "simulator"

    base_instr, base_instr_files = collect_instr(base_root)
    pivot_instr, pivot_instr_files = collect_instr(pivot_root)
    base_code, base_code_files = collect_code(base_root)
    pivot_code, pivot_code_files = collect_code(pivot_root)

    print(f"baseline instruction CSV: {len(base_instr_files)}")
    print(f"pivot instruction CSV: {len(pivot_instr_files)}")
    print(f"baseline code CSV: {len(base_code_files)}")
    print(f"pivot code CSV: {len(pivot_code_files)}")

    print_instr_table("Baseline：最贵 instruction", base_instr, args.limit)
    print_instr_table("Pivot：最贵 instruction", pivot_instr, args.limit)
    print_delta(base_instr, pivot_instr, args.limit)
    print_code_table("Baseline：最贵代码行", base_code, args.limit)
    print_code_table("Pivot：最贵代码行", pivot_code, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
