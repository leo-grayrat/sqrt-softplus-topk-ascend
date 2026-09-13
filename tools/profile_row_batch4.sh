#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "用法: $0 <输出标签，如 Ascend920A> [NPU架构]" >&2
    echo "例如: $0 Ascend920A dav-2201" >&2
    exit 2
fi

SIM_LABEL="$1"
NPU_ARCH="${2:-dav-2201}"
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
WORK_ROOT="${TMPDIR:-/tmp}/${REPO_NAME}-row-batch4-${USER:-user}"
OUT_ROOT="$ROOT/profile-output/${SIM_LABEL}_rowbatch4_e1024_k8"
CONTROL_WT="$WORK_ROOT/bulk-control-wt"
HARNESS_SRC="$ROOT/tools/profile_harness"
SPECIALIZED_HEADER="$ROOT/tools/row_batch4/dense_row_specialized_sequential_profile.h"
BATCH_HEADER="$ROOT/tools/row_batch4/dense_row_batch4_profile.h"
SIM_LIB="$ASCEND_HOME_PATH/tools/simulator/$SIM_ARCH_DIR/lib"

if [[ ! -d "$SIM_LIB" ]]; then
    echo "simulator lib 不存在: $SIM_LIB" >&2
    exit 1
fi
if [[ ! -f "$HARNESS_SRC/CMakeLists.txt" || ! -f "$HARNESS_SRC/main.cpp" ]]; then
    echo "缺少 tools/profile_harness；请在 profile-topk-cost-model 分支运行。" >&2
    exit 1
fi
for required in "$SPECIALIZED_HEADER" "$BATCH_HEADER"; do
    if [[ ! -f "$required" ]]; then
        echo "缺少 profiling header: $required" >&2
        exit 1
    fi
done

mkdir -p "$WORK_ROOT" "$OUT_ROOT"

echo "row-batch4 profiling config: arch=$NPU_ARCH rows=4 E=1024 k=8 blockNum=1"
echo "variants: bulk_control / specialized_sequential / batch4"
echo "simulator libs: $SIM_LIB"

git -C "$ROOT" fetch origin bulk-prefix-pack
if [[ -d "$CONTROL_WT" ]]; then
    git -C "$ROOT" worktree remove --force "$CONTROL_WT" >/dev/null 2>&1 || true
fi
git -C "$ROOT" worktree add --detach "$CONTROL_WT" origin/bulk-prefix-pack

generate_control_source() {
    local out="$1"
    python3 - "$CONTROL_WT/submission/kernel.asc" \
        "$CONTROL_WT/submission/dense_repeat_override.h" "$out" <<'PY'
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

out = (
    '#include <cmath>\n'
    '#include "kernel_operator.h"\n\n'
    'namespace {\n'
    + common
    + prefix
    + '\n'
    + override
    + '\n}  // namespace\n\n'
    + 'void LaunchProfileKernel(\n'
      '    uint32_t blockDim, void* stream, uint8_t* routerLogits, uint8_t* topkScores,\n'
      '    int32_t rows, int32_t experts, int32_t inner, int32_t k, int32_t blockNum)\n'
      '{\n'
      '    sqrt_softplus_prefix_merge_top_k_dispatch_custom<<<blockDim, nullptr, stream>>>(\n'
      '        routerLogits, topkScores, rows, experts, inner, k, blockNum);\n'
      '}\n\n'
      '#include "main.cpp"\n'
)
out_path.write_text(out, encoding="utf-8")
print(f"generated bulk_control: {out_path}")
PY
}

generate_fixed_source() {
    local label="$1"
    local header="$2"
    local kernel_name="$3"
    local out="$4"
    python3 - "$CONTROL_WT/submission/kernel.asc" "$header" "$kernel_name" "$label" "$out" <<'PY'
from pathlib import Path
import sys

kernel_path = Path(sys.argv[1])
header_path = Path(sys.argv[2])
kernel_name = sys.argv[3]
label = sys.argv[4]
out_path = Path(sys.argv[5])
src = kernel_path.read_text(encoding="utf-8")
header = header_path.read_text(encoding="utf-8")

namespace_marker = "namespace {"
namespace_pos = src.find(namespace_marker)
common_end = src.find("// k=1 时无需完整 TopK", namespace_pos)
if min(namespace_pos, common_end) < 0:
    raise SystemExit("无法定位 production 公共段")

common_begin = namespace_pos + len(namespace_marker)
common = src[common_begin:common_end]

out = (
    '#include <cmath>\n'
    '#include "kernel_operator.h"\n\n'
    'namespace {\n'
    + common
    + '\n'
    + header
    + '\n}  // namespace\n\n'
    + 'void LaunchProfileKernel(\n'
      '    uint32_t blockDim, void* stream, uint8_t* routerLogits, uint8_t* topkScores,\n'
      '    int32_t rows, int32_t experts, int32_t inner, int32_t k, int32_t blockNum)\n'
      '{\n'
    + f'    {kernel_name}<<<blockDim, nullptr, stream>>>(\n'
      '        routerLogits, topkScores, rows, experts, inner, k, blockNum);\n'
      '}\n\n'
      '#include "main.cpp"\n'
)
out_path.write_text(out, encoding="utf-8")
print(f"generated {label}: {out_path}")
PY
}

generate_specialized_source() {
    generate_fixed_source \
        specialized_sequential \
        "$SPECIALIZED_HEADER" \
        sqrt_softplus_dense_row_specialized_sequential_profile_custom \
        "$1"
}

generate_batch_source() {
    generate_fixed_source \
        batch4 \
        "$BATCH_HEADER" \
        sqrt_softplus_dense_row_batch4_profile_custom \
        "$1"
}

build_and_profile() {
    local label="$1"
    local generator="$2"
    local harness="$WORK_ROOT/harness-$label"
    local build="$harness/build_sim_profile"
    local output="$OUT_ROOT/$label"
    local log="$OUT_ROOT/${label}_msprof.log"

    rm -rf "$harness" "$output" "$log"
    mkdir -p "$harness"
    cp "$HARNESS_SRC/CMakeLists.txt" "$harness/CMakeLists.txt"
    cp "$HARNESS_SRC/main.cpp" "$harness/main.cpp"
    "$generator" "$harness/kernel_profile.asc"

    cmake -S "$harness" -B "$build" \
        -DCMAKE_ASC_RUN_MODE=sim \
        -DCMAKE_ASC_ARCHITECTURES="$NPU_ARCH"
    cmake --build "$build" -j"$(nproc)" --target topk_profile_one

    echo
    echo "=== profiling $label: rows=4 E=1024 k=8 blockNum=1 ==="
    LD_LIBRARY_PATH="$SIM_LIB:${LD_LIBRARY_PATH:-}" \
    msprof op simulator \
        --output="$output" \
        "$build/topk_profile_one" "4" "1024" "8" "1" | tee "$log"

    if ! grep -q "PASS: single-shape profile driver" "$log"; then
        echo "[FAIL] $label 未通过 host golden correctness" >&2
        exit 1
    fi

    local instr_csv
    instr_csv="$(find "$output" -type f -name '*_instr_exe.csv' -print -quit 2>/dev/null || true)"
    if [[ -z "$instr_csv" ]]; then
        echo "[FAIL] $label 未生成 *_instr_exe.csv" >&2
        exit 1
    fi
    echo "[OK] $label instruction CSV: $instr_csv"
}

build_and_profile bulk_control generate_control_source
build_and_profile specialized_sequential generate_specialized_source
build_and_profile batch4 generate_batch_source

echo
echo "完成。输出目录: $OUT_ROOT"
echo "接下来："
echo "  python3 tools/summarize_row_batch4.py '$OUT_ROOT'"
