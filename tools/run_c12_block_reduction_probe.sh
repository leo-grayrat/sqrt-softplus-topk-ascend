#!/usr/bin/env bash
# Source CANN set_env.sh first. Freeze each variant in a new build directory.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${1:?Usage: run_c12_block_reduction_probe.sh /absolute/new/build}
case "$build" in /*) ;; *) exit 2 ;; esac
test ! -e "$build/src"
mkdir -p "$build/src/debug" "$build/logs" "$build/run"
cp "$0" "$build/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
git -C "$repo" archive deb10330f79908c0f8929d3a5cf236e0fd128eed submission | tar -x -C "$build/src"
for mode in cpu sim; do
 cmake -S "$build/src/debug/cpu_topk_probe" -B "$build/$mode" -DCMAKE_ASC_RUN_MODE="$mode" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$build/logs/$mode-build.log" 2>&1
 cmake --build "$build/$mode" -j2 --target topk_block_reduce_full topk_block_reduce_raw topk_block_reduce_baseline >> "$build/logs/$mode-build.log" 2>&1
done
cd "$build/run"
for kind in raw full baseline; do
 "$build/cpu/topk_block_reduce_$kind" > "$build/logs/$kind-cpu.log" 2>&1
 grep 'PASS: 51' "$build/logs/$kind-cpu.log"
done
for rows in 96 192; do
 for kind in full baseline; do
  msprof op simulator --application="$build/sim/topk_block_reduce_$kind $rows 256 3 1" --soc-version=Ascend910B1 --output="$build/profile-$kind-r$rows" --timeout=5 > "$build/logs/$kind-r$rows.log" 2>&1
  grep -q 'PASS: 1' "$build/logs/$kind-r$rows.log"
  grep -E 'veccore0|PASS:' "$build/logs/$kind-r$rows.log"
 done
done
