#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "用法: $0 <simulator型号，如 Ascend920A> [rows] [E] [k] [NPU架构]" >&2
    echo "例如: $0 Ascend920A 1 1024 8 dav-2201" >&2
    exit 2
fi

SIM_SOC="$1"
ROWS="${2:-1}"
EXPERTS="${3:-1024}"
K="${4:-8}"
NPU_ARCH="${5:-dav-2201}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "未找到 cmake；请先配置 CANN 开发环境。" >&2
    exit 1
fi
if ! command -v msprof >/dev/null 2>&1; then
    echo "未找到 msprof；请先 source CANN set_env.sh。" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH 为空；请先 source CANN set_env.sh。" >&2
    exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "$ROOT")"
WORK_ROOT="${TMPDIR:-/tmp}/${REPO_NAME}-profile-${USER:-user}"
OUT_ROOT="$ROOT/profile-output/${SIM_SOC}_r${ROWS}_e${EXPERTS}_k${K}"
BASE_WT="$WORK_ROOT/baseline"
PIVOT_WT="$WORK_ROOT/pivot"
HARNESS_SRC="$ROOT/tools/profile_harness"

mkdir -p "$WORK_ROOT" "$OUT_ROOT"

echo "profiling config: arch=$NPU_ARCH simulator=$SIM_SOC rows=$ROWS E=$EXPERTS k=$K"

git -C "$ROOT" fetch origin main pivot-topk-rewrite

cleanup_worktree() {
    local path="$1"
    if [[ -d "$path" ]]; then
        git -C "$ROOT" worktree remove --force "$path" >/dev/null 2>&1 || true
    fi
}

cleanup_worktree "$BASE_WT"
cleanup_worktree "$PIVOT_WT"

git -C "$ROOT" worktree add --detach "$BASE_WT" origin/main
git -C "$ROOT" worktree add --detach "$PIVOT_WT" origin/pivot-topk-rewrite

generate_profile_source() {
    local label="$1"
    local source_kernel="$2"
    local override_header="$3"
    local output_kernel="$4"

    python3 - "$label" "$source_kernel" "$override_header" "$output_kernel" <<'PY'
from pathlib import Path
import sys

label = sys.argv[1]
kernel_path = Path(sys.argv[2])
override_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])

src = kernel_path.read_text(encoding="utf-8")
override = override_path.read_text(encoding="utf-8")

# 只抽取 FP32 k<=8 所需的真实 production device 段，避免把 host tiling/graph
# 依赖带入 profiling。算法正文来自待测 worktree，不维护第二份手写实现。
namespace_marker = "namespace {"
namespace_pos = src.find(namespace_marker)
common_end = src.find("// k=1 时无需完整 TopK", namespace_pos)
prefix_begin = src.find("// FP32 小 k：Sort32", common_end)
prefix_end = src.find("// k<=16：", prefix_begin)

if min(namespace_pos, common_end, prefix_begin, prefix_end) < 0:
    raise SystemExit("无法定位 production 公共段或 prefix-merge 段，停止生成 profiling kernel")

common_begin = namespace_pos + len(namespace_marker)
common = src[common_begin:common_end]
prefix = src[prefix_begin:prefix_end]

if label == "baseline":
    kernel_name = "sqrt_softplus_prefix_merge_top_k_dispatch_custom"
elif label == "pivot":
    kernel_name = "sqrt_softplus_pivot_top_k_custom"
else:
    raise SystemExit(f"未知 profiling variant: {label}")

# 使用 CANN 9.0 原生 ASC 单 translation unit：production kernel 保持原来的
# __global__ __vector__ 写法；host wrapper 直接用 <<<>>> launch。这样完全绕过
# legacy ascendc_library 在 AIV auto-gen 阶段对 __global__ 的重定义。
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
      "    uint32_t blockDim,\n"
      "    void* stream,\n"
      "    uint8_t* routerLogits,\n"
      "    uint8_t* topkScores,\n"
      "    int32_t rows,\n"
      "    int32_t experts,\n"
      "    int32_t inner,\n"
      "    int32_t k,\n"
      "    int32_t blockNum)\n"
      "{\n"
    + f"    {kernel_name}<<<blockDim, nullptr, stream>>>(\n"
      "        routerLogits, topkScores, rows, experts, inner, k, blockNum);\n"
      "}\n\n"
      '#include "main.cpp"\n'
)

# 这里故意要求临时源仍保留 production 的 ASC kernel 风格；若以后 production
# 入口发生变化，先人工确认，不让 profiling 脚本静默改写修饰符。
if "__global__ __vector__ void" not in out:
    raise SystemExit("未找到 production __global__ __vector__ kernel 入口")
if "__aicore__ void sqrt_softplus_" in out:
    raise SystemExit("profiling kernel 被意外改写成 legacy __aicore__ 入口")

out_path.write_text(out, encoding="utf-8")
print(f"generated native ASC profile source: {out_path} ({label}, kernel={kernel_name})")
PY
}

build_and_profile() {
    local label="$1"
    local wt="$2"
    local harness="$wt/debug/topk_profile_one"
    local build="$harness/build_sim_profile"
    local output="$OUT_ROOT/$label"
    local override_header

    if [[ "$label" == "baseline" ]]; then
        override_header="$wt/submission/dense_repeat_override.h"
    else
        override_header="$wt/submission/pivot_topk_override.h"
    fi

    rm -rf "$harness" "$output"
    mkdir -p "$harness"

    cp "$HARNESS_SRC/CMakeLists.txt" "$harness/CMakeLists.txt"
    cp "$HARNESS_SRC/main.cpp" "$harness/main.cpp"

    generate_profile_source \
        "$label" \
        "$wt/submission/kernel.asc" \
        "$override_header" \
        "$harness/kernel_profile.asc"

    cmake \
        -S "$harness" \
        -B "$build" \
        -DCMAKE_ASC_RUN_MODE=sim \
        -DCMAKE_ASC_ARCHITECTURES="$NPU_ARCH"

    cmake --build "$build" -j"$(nproc)" --target topk_profile_one

    echo
    echo "=== profiling $label: arch=$NPU_ARCH simulator=$SIM_SOC rows=$ROWS E=$EXPERTS k=$K ==="

    msprof op simulator \
        --soc-version="$SIM_SOC" \
        --output="$output" \
        "$build/topk_profile_one" "$ROWS" "$EXPERTS" "$K"

    local instr_csv
    instr_csv="$(find "$output" -type f -name '*_instr_exe.csv' -print -quit 2>/dev/null || true)"
    if [[ -z "$instr_csv" ]]; then
        echo >&2
        echo "[FAIL] $label profiling 没有生成 *_instr_exe.csv。" >&2
        echo "这表示 simulator/profile 实际未成功；停止，不继续跑另一边。" >&2
        exit 1
    fi
    echo "[OK] $label instruction CSV: $instr_csv"
}

build_and_profile baseline "$BASE_WT"
build_and_profile pivot "$PIVOT_WT"

echo
echo "完成。原始结果："
echo "  baseline: $OUT_ROOT/baseline"
echo "  pivot:    $OUT_ROOT/pivot"
echo
echo "接下来运行："
echo "  python3 tools/summarize_topk_profile.py '$OUT_ROOT' > '$OUT_ROOT/summary.md'"
echo "  cat '$OUT_ROOT/summary.md'"
echo
echo "请不要用 simulator 时长代替 Judge 时长；这里只比较指令和流水结构。"
