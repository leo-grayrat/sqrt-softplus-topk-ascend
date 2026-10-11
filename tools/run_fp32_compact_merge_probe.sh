#!/usr/bin/env bash
# Run after sourcing CANN set_env.sh. Baseline, scratch and candidate use the
# same production-entry probe. Give each mode a distinct absolute build path.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?Usage: run_fp32_compact_merge_probe.sh baseline|scratch|candidate /absolute/build}
build=${2:?Missing build directory}
case "$mode" in baseline|scratch|candidate) ;; *) exit 2 ;; esac
case "$build" in /*) ;; *) echo 'Build directory must be absolute'; exit 2 ;; esac
mkdir -p "$build/src/debug" "$build/run" "$build/logs"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
if [ "$mode" = candidate ]; then
    cp -r "$repo/submission" "$build/src/"
else
    git -C "$repo" archive 1069836ffa0aa6357c12fa5a0c100bea0ce664d5 submission | tar -x -C "$build/src"
    if [ "$mode" = scratch ]; then
        patch -d "$build/src" -p1 < "$repo/debug/fp32_k8_scratch_reuse.patch"
    fi
fi
target=topk_fp32_scratch_production_probe
for run_mode in cpu sim; do
    cmake -S "$build/src/debug/cpu_topk_probe" -B "$build/$run_mode" \
        -DCMAKE_ASC_RUN_MODE="$run_mode" -DCMAKE_ASC_ARCHITECTURES=dav-2201 \
        > "$build/logs/$run_mode-build.log" 2>&1
    cmake --build "$build/$run_mode" -j2 --target "$target" \
        >> "$build/logs/$run_mode-build.log" 2>&1
done
cd "$build/run"
"$build/cpu/$target" > "$build/logs/cpu-run.log" 2>&1
tail -n 1 "$build/logs/cpu-run.log"
for rows in 64 128 1; do
    msprof op simulator --application="$build/sim/$target $rows 210 8 1" \
        --soc-version=Ascend910B1 --output="$build/profile-r$rows" --timeout=5 \
        > "$build/logs/sim-r$rows.log" 2>&1
    grep -E 'core0.veccore0|PASS:' "$build/logs/sim-r$rows.log"
done
if [ "$mode" = candidate ]; then
    msprof op simulator --application="$build/sim/$target 257 210 8 8" \
        --soc-version=Ascend910B1 --output="$build/profile-r257-c8" --timeout=5 \
        > "$build/logs/sim-r257-c8.log" 2>&1
    grep -E 'veccore0|PASS:' "$build/logs/sim-r257-c8.log"
    cmake -S "$build/src/submission" -B "$build/device" -DNPU_ARCH=dav-2201 \
        > "$build/logs/device-build.log" 2>&1
    cmake --build "$build/device" -j1 >> "$build/logs/device-build.log" 2>&1
fi
