#!/usr/bin/env bash
set -eo pipefail
r=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?new absolute build directory}
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp -r "$r/debug/cpu_topk_probe" "$b/src/debug/"
git -C "$r" archive b8bebe9307de2adb503a92bb470ae18b789a9a13 submission | tar -x -C "$b/src"
(
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
for kind in full raw; do
 cmake --build "$b/cpu" -j1 --target topk_c15_batch_filter_$kind >> "$b/logs/cpu-build.log" 2>&1
 cd "$b/run"
 for mode in 1 0; do
  FILTER_INPUT_MODE=$mode "$b/cpu/topk_c15_batch_filter_$kind" > "$b/logs/$kind-cpu-mode$mode.log" 2>&1
  grep 'PASS:' "$b/logs/$kind-cpu-mode$mode.log"
 done
 for mode in 2 3 4 5 6 7 8; do
  FILTER_INPUT_MODE=$mode "$b/cpu/topk_c15_batch_filter_$kind" 23 1024 64 1 > "$b/logs/$kind-extra-mode$mode.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/$kind-extra-mode$mode.log"
 done
done
echo CPU_COMPLETE
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j1 --target topk_c15_batch_filter_full topk_c15_batch_filter_baseline >> "$b/logs/sim-build.log" 2>&1
for mode in 1 0; do
 for kind in baseline full; do
 key=$kind-mode$mode-r64
 mkdir -p "$b/run-$key"
 (cd "$b/run-$key"; FILTER_INPUT_MODE=$mode msprof op simulator --application="$b/sim/topk_c15_batch_filter_$kind 64 1024 64 1" --soc-version=Ascend910B1 --output="$b/profile-$key" --timeout=5) > "$b/logs/$key-sim.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/$key-sim.log"
 grep 'core0.veccore0' "$b/logs/$key-sim.log"
 done
done
echo COMPLETE
) > "$b/driver.log" 2>&1
