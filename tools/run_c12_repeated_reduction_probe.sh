#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${1:?Usage: run_c12_repeated_reduction_probe.sh /absolute/new/build}
case "$build" in /*) ;; *) exit 2 ;; esac
test ! -e "$build/src"
mkdir -p "$build/src/debug" "$build/logs" "$build/run"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
git -C "$repo" archive deb10330f79908c0f8929d3a5cf236e0fd128eed submission | tar -x -C "$build/src"
for mode in cpu sim; do
 cmake -S "$build/src/debug/cpu_topk_probe" -B "$build/$mode" -DCMAKE_ASC_RUN_MODE="$mode" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$build/logs/$mode-build.log" 2>&1
 cmake --build "$build/$mode" -j2 --target topk_reduction_full topk_reduction_raw >> "$build/logs/$mode-build.log" 2>&1
done
cd "$build/run"
for kind in raw full; do
 "$build/cpu/topk_reduction_$kind" > "$build/logs/$kind-cpu.log" 2>&1
 grep 'PASS: 33' "$build/logs/$kind-cpu.log"
done
for rows in 96 192; do
 msprof op simulator --application="$build/sim/topk_reduction_full $rows 256 3 1" --soc-version=Ascend910B1 --output="$build/profile-full-r$rows" --timeout=5 > "$build/logs/full-r$rows.log" 2>&1
 grep -q 'PASS: 1' "$build/logs/full-r$rows.log"
done
