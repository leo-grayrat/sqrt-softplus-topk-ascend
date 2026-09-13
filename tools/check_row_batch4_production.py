#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "submission" / "kernel.asc"
HEADER = ROOT / "submission" / "dense_repeat_override.h"

kernel = KERNEL.read_text(encoding="utf-8")
header = HEADER.read_text(encoding="utf-8")

kernel_required = [
    "useRowBatch4",
    "k32 == 8",
    "experts == 1024",
    "rows % 4 == 0",
    "rowBatchGroupCount",
    "rowBatchBlockNum",
]
for needle in kernel_required:
    if needle not in kernel:
        raise SystemExit(f"FAIL: kernel.asc missing {needle!r}")

header_required = [
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
]
for needle in header_required:
    if needle not in header:
        raise SystemExit(f"FAIL: dense_repeat_override.h missing {needle!r}")

# The batched kernel must issue one source-level Sort32 call for all 4 rows.
start = header.find("class SqrtSoftplusDenseRowBatch4TopKKernel")
end = header.find("__global__ __vector__ void sqrt_softplus_dense_row_batch4_top_k_custom", start)
if start < 0 or end < 0:
    raise SystemExit("FAIL: cannot isolate production batch4 class")
batch_class = header[start:end]
if batch_class.count("AscendC::Sort32<float>(") != 1:
    raise SystemExit("FAIL: production batch4 class must contain exactly one Sort32 call")

print("PASS: production row-batch4 structure is present")
