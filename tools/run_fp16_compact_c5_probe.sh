#!/usr/bin/env bash
# Source CANN set_env.sh first. Keep this script unchanged during execution.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?Usage: run_fp16_compact_c5_probe.sh baseline|candidate /absolute/build}
build=${2:?Missing build directory}
case "$mode" in baseline|candidate) ;; *) exit 2 ;; esac
case "$build" in /*) ;; *) echo 'Build directory must be absolute'; exit 2 ;; esac
mkdir -p "$build/src/debug" "$build/run" "$build/logs"
cp "$0" "$build/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
if [ "$mode" = candidate ]; then
    cp -r "$repo/submission" "$build/src/"
else
    git -C "$repo" archive da7197819a225f703ae4a6409def92676d34a0fd submission | tar -x -C "$build/src"
fi
target=topk_fp16_compact_production_probe
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
for shape in '64 200 16 1' '128 200 16 1' '1 200 16 1' \
             '257 128 16 8' '121 512 10 8'; do
    set -- $shape
    msprof op simulator --application="$build/sim/$target $1 $2 $3 $4" \
        --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-k$3-c$4" --timeout=5 \
        > "$build/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
    grep -E 'veccore0|PASS:' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
    grep -q 'PASS: 1 fp16 production compact cases' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
done
if [ "$mode" = candidate ]; then
    for shape in '257 200 16 8' '513 209 9 8' '65 193 9 1' '65 223 15 1'; do
        set -- $shape
        msprof op simulator --application="$build/sim/$target $1 $2 $3 $4" \
            --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-k$3-c$4" --timeout=5 \
            > "$build/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
        grep -E 'veccore0|PASS:' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
        grep -q 'PASS: 1 fp16 production compact cases' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
    done
    cmake -S "$build/src/submission" -B "$build/device" -DNPU_ARCH=dav-2201 \
        > "$build/logs/device-build.log" 2>&1
    cmake --build "$build/device" -j1 >> "$build/logs/device-build.log" 2>&1
fi
