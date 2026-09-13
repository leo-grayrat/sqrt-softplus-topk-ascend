#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "用法: $0 <输出标签，如 Ascend920A> [rows] [E] [k] [NPU架构]" >&2
    echo "例如: $0 Ascend920A 1 1024 8 dav-2201" >&2
    exit 2
fi

SIM_LABEL="$1"
ROWS="${2:-1}"
EXPERTS="${3:-1024}"
K="${4:-8}"
NPU_ARCH="${5:-dav-2201}"
SIM_ARCH_DIR="${NPU_ARCH//-/_}"

for cmd in cmake msprof python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "未找到 $cmd；请先配置 CANN 开发环境。" >&2
        exit 1
    fi
done
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH 为空；请先 source CANN set_env.sh。" >&2
    exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "$ROOT")"
WORK_ROOT="${TMPDIR:-/tmp}/${REPO_NAME}-bulk-profile-${USER:-user}"
OUT_ROOT="$ROOT/profile-output/${SIM_LABEL}_bulkpack_r${ROWS}_e${EXPERTS}_k${K}"
BASE_WT="$WORK_ROOT/baseline"
BULK_WT="$WORK_ROOT/bulk"
HARNESS_SRC="$ROOT/tools/profile_harness"
SIM_LIB="$ASCEND_HOME_PATH/tools/simulator/$SIM_ARCH_DIR/lib"

if [[ ! -d "$SIM_LIB" ]]; then
    echo "simulator lib 不存在: $SIM_LIB" >&2
    exit 1
fi
if [[ ! -f "$HARNESS_SRC/CMakeLists.txt" || ! -f "$HARNESS_SRC/main.cpp" ]]; then
    echo "缺少 tools/profile_harness；请在 profile-topk-cost-model 分支运行本脚本。" >&2
    exit 1
fi

mkdir -p "$WORK_ROOT" "$OUT_ROOT"

echo "profiling config: arch=$NPU_ARCH simulator_dir=$SIM_ARCH_DIR rows=$ROWS E=$EXPERTS k=$K"
echo "simulator libs: $SIM_LIB"

git -C "$ROOT" fetch origin main bulk-prefix-pack

cleanup_worktree() {
    local path="$1"
    if [[ -d "$path" ]]; then
        git -C "$ROOT" worktree remove --force "$path" >/dev/null 2>&1 || true
    fi
}

cleanup_worktree "$BASE_WT"
cleanup_worktree "$BULK_WT"

git -C "$ROOT" worktree add --detach "$BASE_WT" origin/main
git -C "$ROOT" worktree add --detach "$BULK_WT" origin/bulk-prefix-pack

generate_profile_source() {
    local label="$1"
    local wt="$2"
    local output_kernel="$3"

    python3 - "$label" "$wt/submission/kernel.asc" "$wt/submission/dense_repeat_override.h" "$output_kernel" <<'PY'
from pathlib import Path
import sys

label = sys.argv[1]
kernel_path = Path(sys.argv[2])
override_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])

src = kernel_path.read_text(encoding="utf-8")
override = override_path.read_text(encoding="utf-8")

namespace_marker = "namespace {"
namespace_pos = src.find(namespace_marker)
common_end = src.find("// k=1 时无需完整 TopK", namespace_pos)
prefix_begin = src.find("// FP32 小 k：Sort32", common_end)
prefix_end = src.find("// k<=16：", prefix_begin)
if min(namespace_pos, common_end, prefix_begin, prefix_end) < 0:
    raise SystemExit("无法定位 production 公共段或 prefix-merge 段")

common_begin = namespace_pos + len(namespace_marker)
common = src[common_begin:common_end]
prefix = src[prefix_begin:prefix_end]
kernel_name = "sqrt_softplus_prefix_merge_top_k_dispatch_custom"

out = (
    '#include <cmath>\n'
    '#include "kernel_operator.h"\n\n'
    'namespace {\n'
    + common
    + prefix
    + "\n"
    + override
    + "\n}  // namespace\n\n"
    + "void LaunchProfileKernel(\n"
      "    uint32_t blockDim, void* stream, uint8_t* routerLogits, uint8_t* topkScores,\n"
      "    int32_t rows, int32_t experts, int32_t inner, int32_t k, int32_t blockNum)\n"
      "{\n"
    + f"    {kernel_name}<<<blockDim, nullptr, stream>>>(\n"
      "        routerLogits, topkScores, rows, experts, inner, k, blockNum);\n"
      "}\n\n"
      '#include "main.cpp"\n'
)

if "__global__ __vector__ void" not in out:
    raise SystemExit("未找到 production __global__ __vector__ kernel 入口")
out_path.write_text(out, encoding="utf-8")
print(f"generated {label}: {out_path}")
PY
}

build_and_profile() {
    local label="$1"
    local wt="$2"
    local harness="$wt/debug/topk_bulk_profile"
    local build="$harness/build_sim_profile"
    local output="$OUT_ROOT/$label"

    rm -rf "$harness" "$output"
    mkdir -p "$harness"
    cp "$HARNESS_SRC/CMakeLists.txt" "$harness/CMakeLists.txt"
    cp "$HARNESS_SRC/main.cpp" "$harness/main.cpp"
    generate_profile_source "$label" "$wt" "$harness/kernel_profile.asc"

    cmake -S "$harness" -B "$build" \
        -DCMAKE_ASC_RUN_MODE=sim \
        -DCMAKE_ASC_ARCHITECTURES="$NPU_ARCH"
    cmake --build "$build" -j"$(nproc)" --target topk_profile_one

    echo
    echo "=== profiling $label: rows=$ROWS E=$EXPERTS k=$K ==="
    LD_LIBRARY_PATH="$SIM_LIB:${LD_LIBRARY_PATH:-}" \
    msprof op simulator \
        --output="$output" \
        "$build/topk_profile_one" "$ROWS" "$EXPERTS" "$K"

    local instr_csv
    instr_csv="$(find "$output" -type f -name '*_instr_exe.csv' -print -quit 2>/dev/null || true)"
    if [[ -z "$instr_csv" ]]; then
        echo "[FAIL] $label 未生成 *_instr_exe.csv" >&2
        exit 1
    fi
    echo "[OK] $label instruction CSV: $instr_csv"
}

build_and_profile baseline "$BASE_WT"
build_and_profile bulk "$BULK_WT"

echo
echo "完成。"
echo "  baseline: $OUT_ROOT/baseline"
echo "  bulk:     $OUT_ROOT/bulk"
echo "接下来："
echo "  python3 tools/summarize_bulk_prefix_profile.py '$OUT_ROOT'"
