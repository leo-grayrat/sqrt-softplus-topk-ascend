#!/usr/bin/env bash
# Source the CANN environment before running. Build outside the repository.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
build=${1:?Usage: run_threshold_probe.sh ABS_BUILD_DIR [cpu|sim]}
mode=${2:-cpu}
case "$mode" in cpu|sim) ;; *) echo 'mode must be cpu or sim' >&2; exit 2;; esac
mkdir -p "$build"
build=$(cd "$build" && pwd)
cmake -S "$root/debug/cpu_topk_probe" -B "$build/$mode" \
    -DCMAKE_ASC_RUN_MODE="$mode" -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build "$build/$mode" -j2 --target topk_threshold_compact_probe \
    topk_threshold_full_probe topk_threshold_baseline_probe
mkdir -p "$build/run"
cd "$build/run"
if [ "$mode" = cpu ]; then
    "$build/cpu/topk_threshold_compact_probe"
    for kind in baseline full; do
        env -u THRESHOLD_RANDOM "$build/cpu/topk_threshold_${kind}_probe"
        THRESHOLD_RANDOM=1 "$build/cpu/topk_threshold_${kind}_probe"
    done
else
    for rows in 4 16 64; do
        for kind in baseline full; do
            THRESHOLD_RANDOM=1 msprof op simulator \
                --application="$build/sim/topk_threshold_${kind}_probe $rows 1024 64 1" \
                --soc-version=Ascend910B1 --output="$build/profile-$kind-r$rows" --timeout=5
        done
    done
    for kind in baseline full; do
        env -u THRESHOLD_RANDOM msprof op simulator \
            --application="$build/sim/topk_threshold_${kind}_probe 16 1024 64 1" \
            --soc-version=Ascend910B1 --output="$build/profile-$kind-adversarial" --timeout=5
    done
fi
