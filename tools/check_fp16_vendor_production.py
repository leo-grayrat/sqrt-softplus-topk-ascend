#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "submission" / "dense_repeat_override.h"
KERNEL = ROOT / "submission" / "kernel.asc"
MAIN = ROOT / "submission" / "main.asc"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"forbidden {label}: {needle}")


def main() -> int:
    header = HEADER.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    main_src = MAIN.read_text(encoding="utf-8")

    require(header, "class SqrtSoftplusVendorPerfTopKHalfKernel", "FP16 vendor kernel in Judge-active header")
    require(header, "GlobalTensor<half>", "half GM tensors")
    require(header, "Sort32<float>", "FP32 Sort32 selection")
    require(header, "StableSoftplusSqrt", "batched FP32 transform")
    require(header, "rawHalfOutputBytes", "raw half output byte count")
    require(
        header,
        "#define sqrt_softplus_top_k_custom sqrt_softplus_top_k_dispatch_custom",
        "generic TopK launch override visible to kernel.asc run_kernel",
    )
    require(header, "k >= 2 && k <= 8", "FP16 small-k device gate")
    forbid(header, "AlignUpBytes(", "host-only AlignUpBytes call from aicore header")

    require(kernel, '#include "dense_repeat_override.h"', "Judge-active override include")
    require(kernel, "sqrt_softplus_top_k_custom<half>", "canonical FP16 launch site")

    require(main_src, '#include "kernel.asc"', "local main uses the same Judge-active kernel")
    forbid(main_src, "run_kernel_base", "local-only run_kernel interception")
    forbid(main_src, "fp16_vendor_override.h", "local-only FP16 override include")

    print("PASS: fp16 k<=8 vendor path is wired through kernel.asc Judge hot path")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
