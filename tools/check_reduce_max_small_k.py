#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "dense_repeat_override.h"

text = HEADER.read_text(encoding="utf-8")

required = [
    "SqrtSoftplusRepeatedMaxTopKKernel",
    "AscendC::ReduceMax<float>(",
    "experts_,",
    "true);",
    "reduceOut.GetValue(0)",
    "reduceOut.GetValue(1)",
    "reinterpret_cast<uint32_t*>",
    "srcLocal.SetValue(maxIndex, -INFINITY)",
    "k >= 2 && k <= 4",
    "chunkCount <= 16 || chunkCount == 32",
]

missing = [needle for needle in required if needle not in text]
if missing:
    raise SystemExit("FAIL: missing repeated-max production pieces: " + ", ".join(repr(x) for x in missing))

# Keep the control flow deliberately small: one repeated-max class and one ReduceMax call site.
start = text.find("class SqrtSoftplusRepeatedMaxTopKKernel")
end = text.find("__global__ __vector__ void sqrt_softplus_prefix_merge_top_k_dispatch_custom", start)
if start < 0 or end < 0:
    raise SystemExit("FAIL: cannot isolate repeated-max class")
body = text[start:end]
if body.count("AscendC::ReduceMax<float>(") != 1:
    raise SystemExit("FAIL: repeated-max class should have exactly one source-level ReduceMax call")

print("PASS: small-k repeated-max production structure is present")
