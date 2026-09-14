#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "dense_repeat_override.h"
text = HEADER.read_text(encoding="utf-8")

required = [
    "SqrtSoftplusVendorPerfTopKKernel",
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

missing = [needle for needle in required if needle not in text]
if missing:
    raise SystemExit(
        "FAIL: missing vendor-perf TopK production pieces: "
        + ", ".join(repr(x) for x in missing)
    )

start = text.find("class SqrtSoftplusVendorPerfTopKKernel")
end = text.find("__global__ __vector__ void sqrt_softplus_prefix_merge_top_k_dispatch_custom", start)
if start < 0 or end < 0:
    raise SystemExit("FAIL: cannot isolate vendor-perf kernel")
body = text[start:end]

if body.count("AscendC::Sort32<float>(") != 1:
    raise SystemExit("FAIL: vendor-perf kernel should have one source-level Sort32 call site")
if "for (int32_t row = blockIdx" in body:
    raise SystemExit("FAIL: vendor-perf kernel must operate on row batches, not the old row-at-a-time loop")

print("PASS: vendor-perf TopK production structure is present")
