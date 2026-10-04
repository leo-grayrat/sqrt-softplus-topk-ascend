#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "debug" / "cpu_topk_probe" / "topk_fp16_vendor_probe.asc"
CMAKE = ROOT / "debug" / "cpu_topk_probe" / "CMakeLists.txt"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def main() -> int:
    if not PROBE.exists():
        raise AssertionError(f"missing probe source: {PROBE.relative_to(ROOT)}")

    probe = PROBE.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")

    require(probe, "LocalTensor<half>", "half input tensor")
    require(probe, "AscendC::Cast(floatInput", "half-to-float cast")
    require(probe, "AscendC::Sort32<float>", "FP32 Sort32")
    require(probe, "constexpr int32_t KEEP_K = 8", "fixed top8 prefix")
    require(probe, "AscendC::MrgSort<float>", "FP32 prefix merge")
    require(probe, "BuildValueMask", "dynamic value mask")
    require(probe, "AscendC::GatherMask", "value extraction")
    require(probe, "AscendC::Cast(outputHalf", "FP32-to-half cast")
    require(probe, "RunOne(stream, 256, 3)", "case12-like E256 k3 coverage")
    require(cmake, "topk_fp16_vendor_probe", "CMake target")

    print("PASS: fp16 vendor probe follows half->FP32 vendor selection structure")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
