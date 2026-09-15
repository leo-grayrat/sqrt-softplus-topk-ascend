#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "dense_repeat_override.h"
header = HEADER.read_text(encoding="utf-8")

required = [
    "SqrtSoftplusVendorPerfTopKKernel",
    "SelectVendorBatchCapacity",
    "SelectVendorBatchRows",
    "BuildVendorValueMask",
    "InitValuePattern",
    "ExtractKValues",
    "VendorStableSoftplusSqrtAutoSync",
    "patternBuffer_",
    "activeBlockNum_",
    "VENDOR_KEEP_K = 8",
    "VENDOR_GATHER_REPEAT = 64",
    "TQue<AscendC::TPosition::VECIN, 2>",
    "InitIndexBatch",
    "RearrangeRows",
    "inputBuffer_",
    "CopyTop8Prefixes",
    "MergeFourTop8",
    "MergeTwoFinal32",
    "AscendC::Sort32<float>(",
    "AscendC::MrgSort<float>(",
    "GatherMask(",
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
    "PackTop8Values",
    "gatherBuffer_",
]:
    if forbidden in header:
        raise SystemExit(f"FAIL: stale vendor-perf path remains: {forbidden!r}")

transform_start = header.find("__aicore__ inline void VendorStableSoftplusSqrtAutoSync")
transform_end = header.find("class SqrtSoftplusVendorPerfTopKKernel", transform_start)
if transform_start < 0 or transform_end < 0:
    raise SystemExit("FAIL: cannot isolate round7 vendor transform")
transform_body = header[transform_start:transform_end]

for op in [
    "AscendC::Abs(",
    "AscendC::Muls(",
    "AscendC::Exp(",
    "AscendC::Adds(",
    "AscendC::Ln(",
    "AscendC::Maxs(",
    "AscendC::Add(",
    "AscendC::Sqrt(",
]:
    if op not in transform_body:
        raise SystemExit(f"FAIL: round7 transform lost required operation: {op!r}")

if transform_body.count("PipeBarrier<PIPE_V>();") != 1:
    raise SystemExit(
        "FAIL: round7 transform must keep exactly one explicit final PIPE_V barrier; "
        "intermediate dependencies should be left to BiSheng auto-sync"
    )

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

# Round 6 true-k extraction remains unchanged.
if "pattern.SetValue(0, BuildVendorValueMask(k_))" not in body:
    raise SystemExit("FAIL: value pattern is not initialized from runtime k")
if "params.repeatTimes = static_cast<uint8_t>(rows)" not in body:
    raise SystemExit("FAIL: true-k GatherMask does not repeat across valid rows")
if "params.src0RepeatStride" not in body or "VENDOR_FINAL_PAIR_FLOATS / 8" not in body:
    raise SystemExit("FAIL: true-k GatherMask does not advance by one final-pair row")
if "ExtractKValues(out, finalPairs, validRows)" not in body:
    raise SystemExit("FAIL: ComputeGroup does not use direct true-k extraction")
if "static_cast<uint32_t>(validRows * k_)" not in body:
    raise SystemExit("FAIL: transform still processes padded top-8 instead of validRows*k")

compute = body[body.find("__aicore__ inline void ComputeGroup"):]
if "VendorStableSoftplusSqrtAutoSync(" not in compute:
    raise SystemExit("FAIL: vendor path does not call round7 auto-synced transform")
if "TransformSelected(" in compute:
    raise SystemExit("FAIL: vendor path still calls the globally barriered TransformSelected")
if "if (k_ == VENDOR_KEEP_K)" in compute:
    raise SystemExit("FAIL: stale k=8-only contiguous copyout split remains")
if "for (int32_t row = 0; row < validRows; ++row)" in compute:
    raise SystemExit("FAIL: stale per-row copyout remains after true-k compaction")
if "static_cast<uint64_t>(validRows) * static_cast<uint64_t>(k_) * sizeof(float)" not in compute:
    raise SystemExit("FAIL: packed copyout byte count is not based on validRows*k")
if "packedOutputBytes" not in compute:
    raise SystemExit("FAIL: packed copyout byte count is not normalized to uint32_t")

print("PASS: round7 vendor transform relies on BiSheng PIPE_V auto-sync with one final explicit barrier")
