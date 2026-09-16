#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OVERRIDE = ROOT / "submission" / "dense_repeat_override.h"
MAIN = ROOT / "submission" / "main.asc"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"forbidden {label}: {needle}")


def main() -> int:
    override = OVERRIDE.read_text(encoding="utf-8")
    main_src = MAIN.read_text(encoding="utf-8")

    require(override, "VENDOR_MEDIUM_KEEP_K = 16", "keep-16 width")
    require(
        override,
        "class SqrtSoftplusVendorPerfTopKHalfMediumKernel",
        "FP16 medium-k vendor kernel",
    )
    require(override, "Sort32<float>", "FP32 Sort32 selection")
    require(override, "BuildVendorMediumValueMask", "dynamic true-k value mask")
    require(override, "StableSoftplusSqrt", "batched FP32 transform")
    require(
        override,
        "k >= 9 && k <= 16",
        "FP16 medium-k dispatch gate",
    )
    require(
        override,
        "SqrtSoftplusVendorPerfTopKHalfMediumKernel op;",
        "medium-k device dispatch",
    )

    # Contest implementation must not depend on editing the starter local main.
    forbid(main_src, "fp16_vendor", "vendor override logic in immutable main.asc")
    forbid(main_src, "run_kernel_base", "run_kernel interception in immutable main.asc")

    print("PASS: fp16 9<=k<=16 uses keep-16 vendor path in Judge-active override")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
