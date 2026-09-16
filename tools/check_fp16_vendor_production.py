#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "dense_repeat_override.h"
KERNEL = ROOT / "submission" / "kernel.asc"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def main() -> int:
    header = HEADER.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")

    require(header, "class SqrtSoftplusVendorPerfTopKHalfKernel", "FP16 vendor kernel")
    require(header, "GlobalTensor<half>", "half GM tensors")
    require(header, "Cast(floatInput", "batched half-to-float conversion")
    require(header, "Sort32<float>", "FP32 Sort32 selection")
    require(header, "ExtractKValues", "true-k value extraction")
    require(header, "StableSoftplusSqrt", "batched FP32 transform")
    require(header, "Cast(halfOut", "batched float-to-half conversion")
    require(header, "sqrt_softplus_vendor_fp16_top_k_custom", "FP16 vendor global kernel")

    require(kernel, "dtype == 1 && k32 <= 8", "FP16 small-k dispatch")
    require(kernel, "sqrt_softplus_vendor_fp16_top_k_custom", "FP16 vendor launch")

    print("PASS: fp16 k<=8 production dispatch uses batched half->FP32 vendor selection")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
