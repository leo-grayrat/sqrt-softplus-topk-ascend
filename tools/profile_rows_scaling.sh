#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "用法: $0 <输出标签，如 Ascend920A> [E] [k] [NPU架构]" >&2
    echo "例如: $0 Ascend920A 1024 8 dav-2201" >&2
    exit 2
fi

SIM_LABEL="$1"
EXPERTS="${2:-1024}"
K="${3:-8}"
NPU_ARCH="${4:-dav-2201}"
SIM_ARCH_DIR="${NPU_ARCH//-/_}"
ROWS_LIST=(1 2 4 8)
BLOCK_NUM=1

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
WORK_ROOT="${TMPDIR:-/tmp}/${REPO_NAME}-rows-scaling-${USER:-user}"
BASE_WT="$WORK_ROOT/baseline"
HARNESS="$WORK_ROOT/harness"
BUILD="$HARNESS/build_sim_profile"
OUT_ROOT="$ROOT/profile-output/${SIM_LABEL}_rowscaling_e${EXPERTS}_k${K}"
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

echo "rows-scaling config: arch=$NPU_ARCH simulator_dir=$SIM_ARCH_DIR E=$EXPERTS k=$K blockNum=$BLOCK_NUM rows=${ROWS_LIST[*]}"
echo "simulator libs: $SIM_LIB"

git -C "$ROOT" fetch origin main
if [[ -d "$BASE_WT" ]]; then
    git -C "$ROOT" worktree remove --force "$BASE_WT" >/dev/null 2>&1 || true
fi
git -C "$ROOT" worktree add --detach "$BASE_WT" origin/main

rm -rf "$HARNESS"
mkdir -p "$HARNESS"
cp "$HARNESS_SRC/CMakeLists.txt" "$HARNESS/CMakeLists.txt"
cp "$HARNESS_SRC/main.cpp" "$HARNESS/main.cpp"

python3 - "$BASE_WT/submission/kernel.asc" "$BASE_WT/submission/dense_repeat_override.h" "$HARNESS/kernel_profile.asc" <<'PY'
from pathlib import Path
import sys

kernel_path = Path(sys.argv[1])
override_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

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
print(f"generated: {out_path}")
PY

cmake -S "$HARNESS" -B "$BUILD" \
    -DCMAKE_ASC_RUN_MODE=sim \
    -DCMAKE_ASC_ARCHITECTURES="$NPU_ARCH"
cmake --build "$BUILD" -j"$(nproc)" --target topk_profile_one

for rows in "${ROWS_LIST[@]}"; do
    output="$OUT_ROOT/rows_${rows}"
    rm -rf "$output"
    echo
    echo "=== profiling rows=$rows E=$EXPERTS k=$K blockNum=$BLOCK_NUM ==="
    LD_LIBRARY_PATH="$SIM_LIB:${LD_LIBRARY_PATH:-}" \
    msprof op simulator \
        --output="$output" \
        "$BUILD/topk_profile_one" "$rows" "$EXPERTS" "$K" "$BLOCK_NUM"

    instr_csv="$(find "$output" -type f -name '*_instr_exe.csv' -print -quit 2>/dev/null || true)"
    if [[ -z "$instr_csv" ]]; then
        echo "[FAIL] rows=$rows 未生成 *_instr_exe.csv" >&2
        exit 1
    fi
    echo "[OK] rows=$rows instruction CSV: $instr_csv"
done

echo
echo "完成。输出目录: $OUT_ROOT"
echo "接下来："
echo "  python3 tools/summarize_rows_scaling.py '$OUT_ROOT'"
