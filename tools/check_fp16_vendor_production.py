#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "fp16_vendor_override.h"
MAIN = ROOT / "submission" / "main.asc"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def main() -> int:
    if not HEADER.exists():
        raise AssertionError(f"missing production header: {HEADER.relative_to(ROOT)}")

    header = HEADER.read_text(encoding="utf-8")
    main_src = MAIN.read_text(encoding="utf-8")

    require(header, "class SqrtSoftplusVendorPerfTopKHalfKernel", "FP16 vendor kernel")
    require(header, "GlobalTensor<half>", "half GM tensors")
    require(header, "floatInput,\n                staging", "batched half-to-float conversion")
    require(header, "Sort32<float>", "FP32 Sort32 selection")
    require(header, "ExtractKValues", "true-k value extraction")
    require(header, "StableSoftplusSqrt", "batched FP32 transform")
    require(header, "halfOut,\n            floatOut", "batched float-to-half conversion")
    require(header, "AlignUpBytes(\n            static_cast<uint32_t>(batchRows_ * VENDOR_KEEP_K) * sizeof(half))", "32B half output queue alignment")
    require(header, "dtype == 1 && k32 >= 2 && k32 <= 8", "FP16 small-k gate")
    require(header, "run_kernel_base", "fallback to canonical dispatch")

    require(main_src, "#define run_kernel run_kernel_base", "canonical run_kernel rename")
    require(main_src, "#include \"kernel.asc\"", "canonical kernel include")
    require(main_src, "#undef run_kernel", "run_kernel rename cleanup")
    require(main_src, "#include \"fp16_vendor_override.h\"", "FP16 production override include")

    print("PASS: fp16 k<=8 production override uses batched half->FP32 vendor selection")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
