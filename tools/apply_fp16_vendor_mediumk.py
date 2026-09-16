#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "debug" / "cpu_topk_probe" / "topk_fp16_vendor_medium_probe.asc"
OVERRIDE = ROOT / "submission" / "dense_repeat_override.h"

probe = PROBE.read_text(encoding="utf-8")
override = OVERRIDE.read_text(encoding="utf-8")

production_class = "SqrtSoftplusVendorPerfTopKHalfMediumKernel"
if production_class in override:
    print("already applied")
    raise SystemExit(0)

const_start = probe.index("constexpr int32_t MEDIUM_KEEP_K = 16;")
class_start = probe.index("class Fp16VendorMediumProbeKernel")
kernel_start = probe.index("__global__ __vector__ void topk_fp16_vendor_medium_probe_kernel")

constants = probe[const_start:class_start]
class_code = probe[class_start:kernel_start]

renames = {
    "MEDIUM_KEEP_K": "VENDOR_MEDIUM_KEEP_K",
    "MEDIUM_FIRST_STAGE_COLS": "VENDOR_MEDIUM_FIRST_STAGE_COLS",
    "MEDIUM_MERGED_GROUP_COLS": "VENDOR_MEDIUM_MERGED_GROUP_COLS",
    "MEDIUM_FINAL_VALUES": "VENDOR_MEDIUM_FINAL_VALUES",
    "MEDIUM_FINAL_PAIR_FLOATS": "VENDOR_MEDIUM_FINAL_PAIR_FLOATS",
    "Fp16VendorMediumProbeKernel": production_class,
}
for old, new in renames.items():
    constants = constants.replace(old, new)
    class_code = class_code.replace(old, new)

anchor = "template <typename T>\nstruct VendorDispatchTag {};"
if anchor not in override:
    raise RuntimeError("VendorDispatchTag anchor not found")

override = override.replace(
    anchor,
    "// FP16 9<=k<=16 keep-16 vendor path.\n"
    + constants
    + class_code
    + "\n"
    + anchor,
    1,
)

small_branch = """    if (k >= 2 && k <= 8 && inner <= 256) {
        SqrtSoftplusVendorPerfTopKHalfKernel op;
        op.Init(routerLogits, topkScores, rows, experts, inner, k, blockNum);
        op.Process();
        return;
    }

    SqrtSoftplusTopKKernel<half> op;
"""

medium_branch = """    if (k >= 2 && k <= 8 && inner <= 256) {
        SqrtSoftplusVendorPerfTopKHalfKernel op;
        op.Init(routerLogits, topkScores, rows, experts, inner, k, blockNum);
        op.Process();
        return;
    }

    if (k >= 9 && k <= 16 && inner <= 512) {
        SqrtSoftplusVendorPerfTopKHalfMediumKernel op;
        op.Init(routerLogits, topkScores, rows, experts, inner, k, blockNum);
        op.Process();
        return;
    }

    SqrtSoftplusTopKKernel<half> op;
"""

if small_branch not in override:
    raise RuntimeError("half dispatch anchor not found")
override = override.replace(small_branch, medium_branch, 1)

# Contest invariant: production integration stays inside the fixed submission
# override. main.asc is deliberately not touched.
OVERRIDE.write_text(override, encoding="utf-8")
print("applied FP16 medium-k keep-16 production path")
