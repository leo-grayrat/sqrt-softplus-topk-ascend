#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "dense_repeat_override.h"

header = HEADER.read_text(encoding="utf-8")

required = [
    "SqrtSoftplusDenseRowBatch4TopKKernel",
    "BATCH_ROWS = 4",
    "EXPERTS = 1024",
    "K = 8",
    "TOTAL_CHUNKS = BATCH_ROWS * CHUNKS_PER_ROW",
    "ProcessBatch(firstRow)",
    "AscendC::Sort32<float>",
    "TOTAL_CHUNKS",
    "sqrt_softplus_dense_row_batch4_top_k_custom",
    "useRowBatch4",
    "rows % SqrtSoftplusDenseRowBatch4TopKKernel::BATCH_ROWS == 0",
    "rows >= SqrtSoftplusDenseRowBatch4TopKKernel::BATCH_ROWS * blockNum",
]
for needle in required:
    if needle not in header:
        raise SystemExit(f"FAIL: dense_repeat_override.h missing {needle!r}")

start = header.find("class SqrtSoftplusDenseRowBatch4TopKKernel")
end = header.find("__global__ __vector__ void sqrt_softplus_dense_row_batch4_top_k_custom", start)
if start < 0 or end < 0:
    raise SystemExit("FAIL: cannot isolate production batch4 class")
batch_class = header[start:end]
if batch_class.count("AscendC::Sort32<float>(") != 1:
    raise SystemExit("FAIL: production batch4 class must contain exactly one Sort32 call")

print("PASS: production row-batch4 structure is present")
