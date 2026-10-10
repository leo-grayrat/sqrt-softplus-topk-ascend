#!/usr/bin/env bash
# Source the installed CANN set_env.sh first. Use separate Linux build roots.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?Usage: run_fp32_scratch_probe.sh baseline|candidate /absolute/build [baseline-ref]}
build=${2:?Missing absolute build directory}
baseline_ref=${3:-aa5ae4f14f662c666954f5ea535e029c3645d0ce}
case "$mode" in baseline|candidate) ;; *) exit 2 ;; esac
case "$build" in /*) ;; *) echo 'Build directory must be absolute'; exit 2 ;; esac
mkdir -p "$build/src/debug" "$build/run" "$build/logs"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
git -C "$repo" archive "$baseline_ref" submission | tar -x -C "$build/src"
if [ "$mode" = candidate ]; then
    patch -d "$build/src" -p1 < "$repo/debug/fp32_k8_scratch_reuse.patch"
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
for shape in '64 210' '128 210' '16 888' '32 888' '1 210' '1 888'; do
    set -- $shape
    msprof op simulator --application="$build/sim/$target $1 $2 8 1" \
        --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2" --timeout=5 \
        > "$build/logs/sim-r$1-e$2.log" 2>&1
    grep -E 'core0.veccore0|PASS:' "$build/logs/sim-r$1-e$2.log"
done
if [ "$mode" = candidate ]; then
    cmake -S "$build/src/submission" -B "$build/device" -DNPU_ARCH=dav-2201 \
        > "$build/logs/device-build.log" 2>&1
    cmake --build "$build/device" -j1 >> "$build/logs/device-build.log" 2>&1
fi
