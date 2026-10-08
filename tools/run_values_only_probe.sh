#!/usr/bin/env bash
# Run after sourcing the installed CANN set_env.sh. Keep build/output off NTFS.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${1:?Usage: run_values_only_probe.sh /absolute/build-directory}
mkdir -p "$build/run"
for mode in cpu sim; do
    cmake -S "$repo/debug/cpu_topk_probe" -B "$build/$mode" \
        -DCMAKE_ASC_RUN_MODE="$mode" -DCMAKE_ASC_ARCHITECTURES=dav-2201
    cmake --build "$build/$mode" -j2 --target topk_values_only_probe \
        topk_values_only_baseline_probe topk_values_only_raw_probe
done
cd "$build/run"
for target in topk_values_only_probe topk_values_only_baseline_probe topk_values_only_raw_probe; do
    "$build/cpu/$target" > "$build/$target-cpu.log" 2>&1
    tail -n 1 "$build/$target-cpu.log"
done
# Includes index setup, layout conversion, selection, transform and GM output.
for rows in 62 124; do
    for target in topk_values_only_probe topk_values_only_baseline_probe; do
        msprof op simulator --application="$build/sim/$target $rows 256 3 1" \
            --soc-version=Ascend910B1 --output="$build/profile-$target-r$rows" \
            --timeout=5 > "$build/$target-sim-r$rows.log" 2>&1
        grep -E 'core0.veccore0|PASS:' "$build/$target-sim-r$rows.log"
    done
done
