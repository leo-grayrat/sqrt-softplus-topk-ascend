#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?Usage: run_c13_selection_research.sh /absolute/new/build}
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
git -C "$repo" archive ec0e9621f28446c1439c06cc580fc84b8e6359d6 submission | tar -x -C "$b/src"
for run in cpu sim; do
 cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/$run" -DCMAKE_ASC_RUN_MODE="$run" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/$run-build.log" 2>&1
 cmake --build "$b/$run" -j2 --target topk_c13_selection_full topk_c13_selection_raw topk_c13_selection_block topk_c13_selection_block_raw topk_c13_selection_baseline >> "$b/logs/$run-build.log" 2>&1
 if [ "$run" = cpu ]; then
  cd "$b/run"
  for kind in full raw block block_raw baseline; do
   "$b/cpu/topk_c13_selection_$kind" > "$b/logs/$kind-cpu.log" 2>&1
   grep 'PASS: 19' "$b/logs/$kind-cpu.log"
  done
 fi
done
cd "$b/run"
for rows in 64 128; do
 for kind in full block baseline; do
  msprof op simulator --application="$b/sim/topk_c13_selection_$kind $rows 210 8 1" --soc-version=Ascend910B1 --output="$b/profile-$kind-r$rows" --timeout=5 > "$b/logs/$kind-r$rows.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/$kind-r$rows.log"
  grep -E 'veccore0|PASS:' "$b/logs/$kind-r$rows.log"
 done
done
for shape in '17 1' '129 8'; do
 set -- $shape
 for kind in raw block_raw; do
  msprof op simulator --application="$b/sim/topk_c13_selection_$kind $1 210 8 $2" --soc-version=Ascend910B1 --output="$b/profile-$kind-r$1-c$2" --timeout=5 > "$b/logs/$kind-r$1-c$2.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/$kind-r$1-c$2.log"
 done
done
