#!/usr/bin/env python3
from pathlib import Path

HARNESS = Path(__file__).resolve().parent / "profile_harness" / "main.cpp"
text = HARNESS.read_text(encoding="utf-8")

checks = [
    ("argc != 4 && argc != 5", "profile harness must accept optional blockNum"),
    ("rows E k [blockNum]", "usage must document optional blockNum"),
    ("argc == 5 ? std::atoi(argv[4]) : rows", "blockNum must default to rows"),
    ("blockNum <= 0 || blockNum > rows", "blockNum range must be validated"),
    ("rows, experts, inner, k, blockNum", "LaunchProfileKernel must receive blockNum"),
    ("blockNum=%d", "run banner must print blockNum"),
]

for needle, message in checks:
    if needle not in text:
        raise SystemExit(f"FAIL: {message}")

print("PASS: profile harness exposes and propagates optional blockNum")
