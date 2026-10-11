#!/usr/bin/env bash
# Source CANN set_env.sh first. All inputs are frozen before building.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?Usage: run_fp16_batch_merge_c8_probe.sh baseline|candidate|merge-only /absolute/build}
build=${2:?Missing build directory}
case "$mode" in baseline|candidate|merge-only) ;; *) exit 2 ;; esac
case "$build" in /*) ;; *) exit 2 ;; esac
mkdir -p "$build/src/debug" "$build/run" "$build/logs"
cp "$0" "$build/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
git -C "$repo" archive 2e24d982bd961dee21fa288ae27fee20ca2d7304 submission | tar -x -C "$build/src"
if [ "$mode" = candidate ]; then
    patch -d "$build/src" -p1 < "$repo/debug/fp16_c8_native_sort.patch"
elif [ "$mode" = merge-only ]; then
    patch -d "$build/src" -p1 < "$repo/debug/fp16_c8_batch_merge_only.patch"
fi
target=topk_fp16_large_production_probe
for run_mode in cpu sim; do
    cmake -S "$build/src/debug/cpu_topk_probe" -B "$build/$run_mode" \
        -DCMAKE_ASC_RUN_MODE="$run_mode" -DCMAKE_ASC_ARCHITECTURES=dav-2201 \
        > "$build/logs/$run_mode-build.log" 2>&1
    cmake --build "$build/$run_mode" -j2 --target "$target" \
        >> "$build/logs/$run_mode-build.log" 2>&1
done
cd "$build/run"
"$build/cpu/$target" > "$build/logs/cpu-run.log" 2>&1
grep 'PASS: 45 fp16 production large-k cases' "$build/logs/cpu-run.log"
for shape in '64 256 64 1' '128 256 64 1' '1 256 64 1' \
             '257 241 64 8' '257 128 16 8' '4096 256 64 32'; do
    set -- $shape
    msprof op simulator --application="$build/sim/$target $1 $2 $3 $4" \
        --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-k$3-c$4" --timeout=5 \
        > "$build/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
    grep -E 'veccore0|PASS:' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
    grep -q 'PASS: 1 fp16 production large-k cases' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
done
if [ "$mode" != baseline ]; then
    for shape in '33 225 64 1' '65 255 64 1' '513 241 64 8'; do
        set -- $shape
        msprof op simulator --application="$build/sim/$target $1 $2 $3 $4" \
            --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-k$3-c$4" --timeout=5 \
            > "$build/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
        grep -E 'veccore0|PASS:' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
        grep -q 'PASS: 1 fp16 production large-k cases' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
    done
    cmake -S "$build/src/submission" -B "$build/device" -DNPU_ARCH=dav-2201 \
        > "$build/logs/device-build.log" 2>&1
    cmake --build "$build/device" -j1 >> "$build/logs/device-build.log" 2>&1
fi
