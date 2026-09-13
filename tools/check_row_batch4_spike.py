#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_HEADER = ROOT / "tools" / "row_batch4" / "dense_row_batch4_profile.h"
SPECIALIZED_HEADER = (
    ROOT / "tools" / "row_batch4" / "dense_row_specialized_sequential_profile.h"
)
SCRIPT = ROOT / "tools" / "profile_row_batch4.sh"

for path in (BATCH_HEADER, SPECIALIZED_HEADER, SCRIPT):
    if not path.exists():
        raise SystemExit(f"FAIL: missing {path.relative_to(ROOT)}")

batch = BATCH_HEADER.read_text(encoding="utf-8")
for needle in [
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
]:
    if needle not in batch:
        raise SystemExit(f"FAIL: batch4 header missing {needle!r}")

if batch.count("AscendC::Sort32<float>(") != 1:
    raise SystemExit("FAIL: batch4 must issue exactly one Sort32 call in source")

specialized = SPECIALIZED_HEADER.read_text(encoding="utf-8")
for needle in [
    "SqrtSoftplusDenseRowSpecializedSequentialProfileKernel",
    "ROWS = 4",
    "EXPERTS = 1024",
    "K = 8",
    "for (int32_t row = 0; row < ROWS; ++row)",
    "ProcessRow(row)",
    "CHUNK_COUNT",
    "sqrt_softplus_dense_row_specialized_sequential_profile_custom",
]:
    if needle not in specialized:
        raise SystemExit(f"FAIL: specialized sequential header missing {needle!r}")

script = SCRIPT.read_text(encoding="utf-8")
for needle in [
    "bulk-prefix-pack",
    "bulk_control",
    "specialized_sequential",
    "batch4",
    '"4" "1024" "8" "1"',
]:
    if needle not in script:
        raise SystemExit(f"FAIL: profile script missing {needle!r}")

print("PASS: row-batch4 attribution spike structure is present")
