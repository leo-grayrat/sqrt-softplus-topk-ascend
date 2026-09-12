#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "用法: $0 <Ascend具体型号，如 Ascend910B1> [rows] [E] [k]" >&2
    exit 2
fi

SOC_VERSION="$1"
ROWS="${2:-1}"
EXPERTS="${3:-1024}"
K="${4:-8}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "未找到 cmake；请先配置 CANN 开发环境。" >&2
    exit 1
fi
if ! command -v msprof >/dev/null 2>&1; then
    echo "未找到 msprof；请先 source CANN set_env.sh。" >&2
    exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
WORK_ROOT="$ROOT/.profile-worktrees"
OUT_ROOT="$ROOT/profile-output/${SOC_VERSION}_r${ROWS}_e${EXPERTS}_k${K}"
BASE_WT="$WORK_ROOT/baseline"
PIVOT_WT="$WORK_ROOT/pivot"

mkdir -p "$WORK_ROOT" "$OUT_ROOT"

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

build_and_profile() {
    local label="$1"
    local wt="$2"
    local target="$3"
    local build="$wt/debug/cpu_topk_probe/build_sim_profile"
    local output="$OUT_ROOT/$label"

    rm -rf "$build" "$output"

    cmake \
        -S "$wt/debug/cpu_topk_probe" \
        -B "$build" \
        -DCMAKE_ASC_RUN_MODE=sim \
        -DCMAKE_ASC_ARCHITECTURES=dav-2201

    cmake --build "$build" -j"$(nproc)" --target "$target"

    echo
    echo "=== profiling $label: rows=$ROWS E=$EXPERTS k=$K ==="
    msprof op simulator \
        --soc-version="$SOC_VERSION" \
        --output="$output" \
        "$build/$target" "$ROWS" "$EXPERTS" "$K"
}

build_and_profile baseline "$BASE_WT" topk_prefix_production_probe
build_and_profile pivot "$PIVOT_WT" topk_pivot_production_probe

echo
echo "完成。原始结果："
echo "  baseline: $OUT_ROOT/baseline"
echo "  pivot:    $OUT_ROOT/pivot"
echo
echo "重点文件："
echo "  */OPPROF_*/simulator/core0.veccore0/*_instr_exe.csv"
echo "  */OPPROF_*/simulator/core0.veccore0/*_code_exe.csv"
echo "  */OPPROF_*/simulator/core0.veccore0/trace.json"
echo
echo "请不要用这次 simulator 时长代替 Judge 时长；这里只比较指令/流水结构。"
