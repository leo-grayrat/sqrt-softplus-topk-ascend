#!/usr/bin/env bash
# Source CANN set_env.sh first. Both modes use the same production-entry probe.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?Usage: run_fp32_compact_c3_probe.sh baseline|candidate /absolute/build}
build=${2:?Missing build directory}
case "$mode" in baseline|candidate) ;; *) exit 2 ;; esac
case "$build" in /*) ;; *) echo 'Build directory must be absolute'; exit 2 ;; esac
mkdir -p "$build/src/debug" "$build/run" "$build/logs"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
if [ "$mode" = candidate ]; then
    cp -r "$repo/submission" "$build/src/"
else
    git -C "$repo" archive 6a4ca0ee30f1d76c8288144bf579ea3285a46397 submission | tar -x -C "$build/src"
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
for rows in 16 32 1; do
    msprof op simulator --application="$build/sim/$target $rows 888 8 1" \
        --soc-version=Ascend910B1 --output="$build/profile-r$rows" --timeout=5 \
        > "$build/logs/sim-r$rows.log" 2>&1
    grep -E 'core0.veccore0|PASS:' "$build/logs/sim-r$rows.log"
done
if [ "$mode" = baseline ]; then
    msprof op simulator --application="$build/sim/$target 257 210 8 8" \
        --soc-version=Ascend910B1 --output="$build/profile-r257-e210-c8" --timeout=5 \
        > "$build/logs/sim-r257-e210-c8.log" 2>&1
    grep -E 'veccore0|PASS:' "$build/logs/sim-r257-e210-c8.log"
fi
if [ "$mode" = candidate ]; then
    for shape in '129 888 8' '65 865 8' '65 896 8' '257 210 8'; do
        set -- $shape
        msprof op simulator --application="$build/sim/$target $1 $2 8 $3" \
            --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-c$3" --timeout=5 \
            > "$build/logs/sim-r$1-e$2-c$3.log" 2>&1
        grep -E 'veccore0|PASS:' "$build/logs/sim-r$1-e$2-c$3.log"
    done
    cmake -S "$build/src/submission" -B "$build/device" -DNPU_ARCH=dav-2201 \
        > "$build/logs/device-build.log" 2>&1
    cmake --build "$build/device" -j1 >> "$build/logs/device-build.log" 2>&1
fi
