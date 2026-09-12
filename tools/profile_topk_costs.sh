#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "用法: $0 <Ascend具体型号，如 Ascend920A> [rows] [E] [k]" >&2
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
REPO_NAME="$(basename "$ROOT")"
WORK_ROOT="${TMPDIR:-/tmp}/${REPO_NAME}-profile-${USER:-user}"
OUT_ROOT="$ROOT/profile-output/${SOC_VERSION}_r${ROWS}_e${EXPERTS}_k${K}"
BASE_WT="$WORK_ROOT/baseline"
PIVOT_WT="$WORK_ROOT/pivot"
HARNESS_SRC="$ROOT/tools/profile_harness"

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
    local harness="$wt/debug/topk_profile_one"
    local build="$harness/build_sim_profile"
    local output="$OUT_ROOT/$label"

    rm -rf "$harness" "$output"
    mkdir -p "$harness"
    cp "$HARNESS_SRC/CMakeLists.txt" "$harness/CMakeLists.txt"
    cp "$HARNESS_SRC/topk_profile_one.asc" "$harness/topk_profile_one.asc"

    cmake \
        -S "$harness" \
        -B "$build" \
        -DCMAKE_ASC_RUN_MODE=sim \
        -DCMAKE_ASC_ARCHITECTURES=dav-2201

    cmake --build "$build" -j"$(nproc)" --target topk_profile_one

    echo
    echo "=== profiling $label: rows=$ROWS E=$EXPERTS k=$K ==="
    msprof op simulator \
        --soc-version="$SOC_VERSION" \
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
