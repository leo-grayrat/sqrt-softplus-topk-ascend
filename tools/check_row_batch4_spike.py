#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "tools" / "row_batch4" / "dense_row_batch4_profile.h"
SCRIPT = ROOT / "tools" / "profile_row_batch4.sh"

if not HEADER.exists():
    raise SystemExit(f"FAIL: missing {HEADER.relative_to(ROOT)}")

text = HEADER.read_text(encoding="utf-8")
required = [
    "SqrtSoftplusDenseRowBatch4ProfileKernel",
    "BATCH_ROWS = 4",
    "EXPERTS = 1024",
    "K = 8",
    "TOTAL_CHUNKS = BATCH_ROWS * CHUNKS_PER_ROW",
    "AscendC::Sort32<float>",
    "TOTAL_CHUNKS",
    "MergeRepeated",
    "128",
    "32",
    "8",
    "AscendC::DataCopyParams",
    "sqrt_softplus_dense_row_batch4_profile_custom",
]
for needle in required:
    if needle not in text:
        raise SystemExit(f"FAIL: batch4 header missing {needle!r}")

# The whole point of the spike is that Sort32 is issued once for all four rows.
if text.count("AscendC::Sort32<float>(") != 1:
    raise SystemExit("FAIL: batch4 must issue exactly one Sort32 call")

if not SCRIPT.exists():
    raise SystemExit(f"FAIL: missing {SCRIPT.relative_to(ROOT)}")

script = SCRIPT.read_text(encoding="utf-8")
for needle in [
    "bulk-prefix-pack",
    "bulk_control",
    "batch4",
    '"4" "1024" "8" "1"',
]:
    if needle not in script:
        raise SystemExit(f"FAIL: profile script missing {needle!r}")

print("PASS: row-batch4 spike structure is present")
