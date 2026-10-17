#!/usr/bin/env bash
set -eo pipefail
r=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?new absolute output directory}
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp -r "$r/debug/cpu_topk_probe" "$b/src/debug/"
git -C "$r" archive b8bebe9307de2adb503a92bb470ae18b789a9a13 submission | tar -x -C "$b/src"
cp "$0" "$b/driver-snapshot.sh"
(
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
for kind in full raw; do
 cmake --build "$b/cpu" -j1 --target topk_c15_batch_filter_$kind >> "$b/logs/cpu-build.log" 2>&1
 for mode in 0 1 2 3 4 5 6 7 8; do
  (cd "$b/run"; FILTER_INPUT_MODE=$mode "$b/cpu/topk_c15_batch_filter_$kind") > "$b/logs/$kind-mode$mode-cpu.log" 2>&1
  grep 'PASS:' "$b/logs/$kind-mode$mode-cpu.log"
 done
done
echo CPU_COMPLETE
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j1 --target topk_c15_batch_filter_full topk_c15_batch_filter_baseline >> "$b/logs/sim-build.log" 2>&1
profile() {
 local kind=$1 mode=$2 rows=$3 cores=$4 key=$1-mode$2-r$3-c$4
 mkdir -p "$b/run-$key"
 (cd "$b/run-$key"; FILTER_INPUT_MODE=$mode msprof op simulator --application="$b/sim/topk_c15_batch_filter_$kind $rows 1024 64 $cores" --soc-version=Ascend910B1 --output="$b/profile-$key" --timeout=8) > "$b/logs/$key-sim.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/$key-sim.log"
}
for shape in '1 64 1' '2 64 1' '0 64 1' '3 64 1' '2 1043 32'; do
 set -- $shape
 profile baseline "$1" "$2" "$3" & left=$!
 profile full "$1" "$2" "$3" & right=$!
 wait "$left"; wait "$right"
done
echo COMPLETE
) > "$b/driver.log" 2>&1
