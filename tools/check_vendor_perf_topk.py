#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "dense_repeat_override.h"
KERNEL = ROOT / "submission" / "kernel.asc"
header = HEADER.read_text(encoding="utf-8")
kernel = KERNEL.read_text(encoding="utf-8")

required_header = [
    "SqrtSoftplusVendorPerfTopKKernel",
    "SelectVendorBatchCapacity",
    "SelectVendorBatchRows",
    "VENDOR_KEEP_K = 8",
    "TQue<AscendC::TPosition::VECIN, 2>",
    "InitIndexBatch",
    "CopyTop8Prefixes",
    "MergeFourTop8",
    "MergeTwoFinal32",
    "AscendC::Sort32<float>(",
    "AscendC::MrgSort<float>(",
    "AscendC::GatherMask(",
    "experts % 32 == 0",
    "k >= 2 && k <= 8",
]

missing = [needle for needle in required_header if needle not in header]
if missing:
    raise SystemExit(
        "FAIL: missing adaptive vendor-perf pieces: "
        + ", ".join(repr(x) for x in missing)
    )

if "rows >= blockNum * vendorBatchRows" in header:
    raise SystemExit("FAIL: old full-batch admission gate is still present")

start = header.find("class SqrtSoftplusVendorPerfTopKKernel")
end = header.find("__global__ __vector__ void sqrt_softplus_prefix_merge_top_k_dispatch_custom", start)
if start < 0 or end < 0:
    raise SystemExit("FAIL: cannot isolate vendor-perf kernel")
body = header[start:end]

if body.count("AscendC::Sort32<float>(") != 1:
    raise SystemExit("FAIL: vendor-perf kernel should have one source-level Sort32 call site")
if "for (int32_t row = blockIdx" in body:
    raise SystemExit("FAIL: vendor-perf kernel must operate on row batches, not the old row-at-a-time loop")

required_kernel = [
    "vendorCapacity",
    "vendorRowsPerCore",
    "vendorBatchRows",
    "vendorGroupCount",
    "vendorBlockNum",
]
missing_kernel = [needle for needle in required_kernel if needle not in kernel]
if missing_kernel:
    raise SystemExit(
        "FAIL: host launch is not adaptive yet: "
        + ", ".join(repr(x) for x in missing_kernel)
    )

print("PASS: adaptive vendor-perf TopK production structure is present")
