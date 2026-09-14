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
    "RearrangeRows",
    "inputBuffer_",
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
        "FAIL: missing staged vendor-perf pieces: "
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
if "SelectVendorBatchRows(rows_, blockNum_, inner_)" not in body:
    raise SystemExit("FAIL: batch capacity is not based on padded inner_")

copy_start = body.find("__aicore__ inline void CopyInGroup")
copy_end = body.find("__aicore__ inline void RearrangeRows", copy_start)
if copy_start < 0 or copy_end < 0:
    raise SystemExit("FAIL: cannot isolate staged CopyInGroup/RearrangeRows")
copy_body = body[copy_start:copy_end]
if "for (int32_t slot = 0; slot < batchRows_; ++slot)" in copy_body:
    raise SystemExit("FAIL: CopyInGroup still issues one DataCopyPad per row")
if "blockCount = static_cast<uint16_t>(validRows)" not in copy_body:
    raise SystemExit("FAIL: CopyInGroup does not issue one multi-row DataCopyPad")

if "AscendC::LocalTensor<float> input = staging;" not in body:
    raise SystemExit("FAIL: aligned rows do not sort directly from staging")
if "input = inputBuffer_.Get<float>();" not in body:
    raise SystemExit("FAIL: nonaligned rows do not switch to rearranged input buffer")
if "validRows * inner_ / VENDOR_SORT_UNIT" not in body:
    raise SystemExit("FAIL: Sort32 still processes dummy rows in the tail batch")
if "BuildFinalPairs(pairA, pairB, validRows)" not in body:
    raise SystemExit("FAIL: merge tree still processes dummy rows in the tail batch")
if "static_cast<uint8_t>(validRows)" not in body:
    raise SystemExit("FAIL: GatherMask still processes dummy rows in the tail batch")

print("PASS: staged adaptive vendor-perf TopK structure is present")
