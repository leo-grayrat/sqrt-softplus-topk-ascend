#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "用法: $0 <simulator型号，如 Ascend920A> [rows] [E] [k] [编译SoC]" >&2
    echo "例如: $0 Ascend920A 1 1024 8 ascend910b1" >&2
    exit 2
fi

SIM_SOC="$1"
ROWS="${2:-1}"
EXPERTS="${3:-1024}"
K="${4:-8}"

# msprof simulator 使用安装目录里的 simulator 名称；ascendc_library 的
# SOC_VERSION 则使用编译器支持的具体 SoC 名称。二者不是同一套标识。
if [[ $# -ge 5 ]]; then
    COMPILE_SOC="$5"
else
    case "$SIM_SOC" in
        Ascend920A)
            COMPILE_SOC="ascend910b1"
            ;;
        *)
            echo "未定义 simulator '$SIM_SOC' 到编译 SoC 的默认映射。" >&2
            echo "请显式提供第 5 个参数，例如 ascend910b1。" >&2
            exit 2
            ;;
    esac
fi

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

echo "profiling config: compile_soc=$COMPILE_SOC simulator=$SIM_SOC rows=$ROWS E=$EXPERTS k=$K"

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

generate_device_source() {
    local source_kernel="$1"
    local override_header="$2"
    local output_kernel="$3"

    python3 - "$source_kernel" "$override_header" "$output_kernel" <<'PY'
from pathlib import Path
import sys

kernel_path = Path(sys.argv[1])
override_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

src = kernel_path.read_text(encoding="utf-8")
override = override_path.read_text(encoding="utf-8")

# profiling 只需要 FP32 k<=8 的真实 production selector。不要把 host-only
# tiling/graph 头和不相关的 TopK/Hierarchical kernel 一起送进 device precompile：
# kernel_type.h 的 DT_* 宏会与 graph/types.h 的同名枚举直接冲突。
namespace_marker = "namespace {"
namespace_pos = src.find(namespace_marker)
common_end = src.find("// k=1 时无需完整 TopK", namespace_pos)
prefix_begin = src.find("// FP32 小 k：Sort32", common_end)
prefix_end = src.find("// k<=16：", prefix_begin)

if min(namespace_pos, common_end, prefix_begin, prefix_end) < 0:
    raise SystemExit("无法定位 production 公共段或 prefix-merge 段，停止生成 profiling kernel")

# production 文件把 device helper / kernel 放在匿名 namespace 中，这对原始 .asc
# 直接编译可工作；ascendc_library 则会自动生成可外部调用的 *_do stub，并在
# auto_gen 层重命名原 kernel，因此 profiling 副本必须让 kernel 具有外部链接。
# 这里只去掉匿名 namespace 外壳，内部类/常量/算法正文保持逐段同源。
common_begin = namespace_pos + len(namespace_marker)
common = src[common_begin:common_end]
prefix = src[prefix_begin:prefix_end]

out = (
    '#include <cmath>\n'
    '#include "kernel_operator.h"\n\n'
    + common
    + prefix
    + "\n"
    + override
    + "\n"
)

# CANN kernel-launch / ascendc_library 的 kernel 入口需要外部链接。官方示例
# 使用 extern "C" __global__ __aicore__，纯 Vector kernel 可额外用 __vector__
# 标记。production .asc 当前是 `__global__ __vector__ void`；只在临时副本中
# 加上 C linkage 与 __aicore__，不修改 Judge 源。
old_entry = "__global__ __vector__ void"
entry_count = out.count(old_entry)
if entry_count == 0:
    raise SystemExit("未找到 __global__ __vector__ kernel 入口，停止生成 profiling kernel")
out = out.replace(
    old_entry,
    'extern "C" __global__ __vector__ __aicore__ void')
if old_entry in out:
    raise SystemExit("profiling kernel 入口规范化不完整")
if "namespace {" in out:
    raise SystemExit("profiling device 源仍残留匿名 namespace")

out_path.write_text(out, encoding="utf-8")
print(
    f"generated {out_path}: exported and normalized {entry_count} vector kernel "
    f"entr{'y' if entry_count == 1 else 'ies'}")
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

    # CANN 9.0 legacy ascendc precompile 的模板提取阶段只识别常规 C/C++ 源；
    # 仓库 production 仍保持 .asc，这里的临时 device-only 副本使用 .cpp。
    # 源码内容由真实 production 公共段 + prefix kernel + 对应 override 自动拼接，
    # 不复制 host tiling 依赖，也不维护第二份手写算法。
    generate_device_source \
        "$wt/submission/kernel.asc" \
        "$override_header" \
        "$harness/kernel_profile.cpp"

    cmake \
        -S "$harness" \
        -B "$build" \
        -DRUN_MODE=sim \
        -DSOC_VERSION="$COMPILE_SOC" \
        -DPROFILE_VARIANT="$label" \
        -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"

    cmake --build "$build" -j"$(nproc)" --target topk_profile_one

    echo
    echo "=== profiling $label: compile=$COMPILE_SOC simulator=$SIM_SOC rows=$ROWS E=$EXPERTS k=$K ==="

    local sim_lib="$ASCEND_HOME_PATH/tools/simulator/$SIM_SOC/lib"
    if [[ ! -d "$sim_lib" ]]; then
        echo "[FAIL] simulator lib 不存在: $sim_lib" >&2
        exit 1
    fi

    LD_LIBRARY_PATH="$sim_lib:${LD_LIBRARY_PATH:-}" \
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
