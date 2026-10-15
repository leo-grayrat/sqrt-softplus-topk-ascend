#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?Usage: run_c14_research_probe.sh /absolute/new/build}
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
# Both full/raw research and baseline use the same pre-C14 production sources.
git -C "$repo" archive 5e48b1998b0e5d987742d3ec420b0f389935a779 submission | tar -x -C "$b/src"
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
cmake --build "$b/cpu" -j2 --target topk_c14_block_full topk_c14_block_raw topk_c14_block_baseline >> "$b/logs/cpu-build.log" 2>&1
cd "$b/run"
for kind in raw full baseline; do
 "$b/cpu/topk_c14_block_$kind" > "$b/logs/$kind-cpu.log" 2>&1
 grep 'PASS: 51' "$b/logs/$kind-cpu.log"
done
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j2 --target topk_c14_block_full topk_c14_block_raw topk_c14_block_baseline >> "$b/logs/sim-build.log" 2>&1
for rows in 64 128; do
 for kind in full baseline; do
  msprof op simulator --application="$b/sim/topk_c14_block_$kind $rows 512 10 1" --soc-version=Ascend910B1 --output="$b/profile-$kind-r$rows" --timeout=5 > "$b/logs/$kind-r$rows.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/$kind-r$rows.log"
  grep -E 'veccore0|PASS:' "$b/logs/$kind-r$rows.log"
 done
done
