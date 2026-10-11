#!/usr/bin/env bash
# Source CANN set_env.sh first. Keep this script unchanged during execution.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?Usage: run_fp32_large_scratch_probe.sh baseline|candidate /absolute/build}
build=${2:?Missing build directory}
case "$mode" in baseline|candidate) ;; *) exit 2 ;; esac
case "$build" in /*) ;; *) echo 'Build directory must be absolute'; exit 2 ;; esac
mkdir -p "$build/src/debug" "$build/run" "$build/logs"
cp "$0" "$build/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
git -C "$repo" archive da7197819a225f703ae4a6409def92676d34a0fd submission | tar -x -C "$build/src"
if [ "$mode" = candidate ]; then
    patch -d "$build/src" -p1 < "$repo/debug/fp32_large_scratch_reuse.patch"
fi
target=topk_fp32_large_production_probe
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
for shape in '28 1024 64 1' '56 1024 64 1' '1 1024 64 1' '33 97 32 8'; do
    set -- $shape
    msprof op simulator --application="$build/sim/$target $1 $2 $3 $4" \
        --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-k$3-c$4" --timeout=5 \
        > "$build/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
    grep -E 'veccore0|PASS:' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
    grep -q 'PASS: 1 fp32 production large-k cases' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
done
if [ "$mode" = candidate ]; then
    for shape in '57 1024 64 8' '113 1024 64 8' '57 993 64 8' '15 1017 64 1' \
                 '9 992 64 8' '9 1024 63 8'; do
        set -- $shape
        msprof op simulator --application="$build/sim/$target $1 $2 $3 $4" \
            --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-k$3-c$4" --timeout=5 \
            > "$build/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
        grep -E 'veccore0|PASS:' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
        grep -q 'PASS: 1 fp32 production large-k cases' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
    done
    cmake -S "$build/src/submission" -B "$build/device" -DNPU_ARCH=dav-2201 \
        > "$build/logs/device-build.log" 2>&1
    cmake --build "$build/device" -j1 >> "$build/logs/device-build.log" 2>&1
fi
