#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?new absolute build directory}
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
cp -r "$repo/submission" "$b/src/"
cd "$b/run"
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
for kind in half_raw half; do
 cmake --build "$b/cpu" -j1 --target "topk_c14_ten_$kind" >> "$b/logs/cpu-build.log" 2>&1
 "$b/cpu/topk_c14_ten_$kind" > "$b/logs/$kind-cpu.log" 2>&1
 grep 'PASS: 17' "$b/logs/$kind-cpu.log"
done
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j1 --target topk_c14_ten_half topk_c14_ten_half_raw >> "$b/logs/sim-build.log" 2>&1
for rows in 64 128; do
 for kind in half half_raw; do
  msprof op simulator --application="$b/sim/topk_c14_ten_$kind $rows 512 10 1" --soc-version=Ascend910B1 --output="$b/profile-$kind-r$rows" --timeout=5 > "$b/logs/$kind-r$rows.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/$kind-r$rows.log"
  grep -E 'veccore0|PASS:' "$b/logs/$kind-r$rows.log"
 done
done
echo COMPLETE
