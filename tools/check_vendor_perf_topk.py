#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "dense_repeat_override.h"
header = HEADER.read_text(encoding="utf-8")

required = [
    "SqrtSoftplusVendorPerfTopKKernel",
    "SelectVendorBatchCapacity",
    "SelectVendorBatchRows",
    "activeBlockNum_",
    "VENDOR_KEEP_K = 8",
    "TQue<AscendC::TPosition::VECIN, 2>",
    "InitIndexBatch",
    "CopyTop8Prefixes",
    "MergeFourTop8",
    "MergeTwoFinal32",
    "AscendC::Sort32<float>(",
    "AscendC::MrgSort<float>(",
    "AscendC::GatherMask(",
    "inner_",
    "copyAlignedElems",
    "negativeInf",
    "k >= 2 && k <= 8",
]

missing = [needle for needle in required if needle not in header]
if missing:
    raise SystemExit(
        "FAIL: missing padded vendor-perf pieces: "
        + ", ".join(repr(x) for x in missing)
    )

for forbidden in [
    "rows >= blockNum * vendorBatchRows",
    "experts % 32 == 0",
    "experts == inner &&",
]:
    if forbidden in header:
        raise SystemExit(f"FAIL: stale vendor-perf admission restriction remains: {forbidden!r}")

start = header.find("class SqrtSoftplusVendorPerfTopKKernel")
end = header.find("__global__ __vector__ void sqrt_softplus_prefix_merge_top_k_dispatch_custom", start)
if start < 0 or end < 0:
    raise SystemExit("FAIL: cannot isolate vendor-perf kernel")
body = header[start:end]

if body.count("AscendC::Sort32<float>(") != 1:
    raise SystemExit("FAIL: vendor-perf kernel should have one source-level Sort32 call site")
if "for (int32_t row = blockIdx" in body:
    raise SystemExit("FAIL: vendor-perf kernel must operate on row batches, not the old row-at-a-time loop")
if "group + activeBlockNum_" not in body:
    raise SystemExit("FAIL: group stride still uses launched block count instead of active block count")
if "slot * inner_" not in body:
    raise SystemExit("FAIL: UB row stride is not padded inner_")
if "batchRows_ * inner_ / VENDOR_SORT_UNIT" not in body:
    raise SystemExit("FAIL: Sort32 repeat count is not based on padded inner_")
if "SelectVendorBatchRows(rows_, blockNum_, inner_)" not in body:
    raise SystemExit("FAIL: batch capacity is not based on padded inner_")

print("PASS: padded adaptive vendor-perf TopK production structure is present")
