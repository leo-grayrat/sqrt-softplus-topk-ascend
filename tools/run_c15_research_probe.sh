#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?Usage: run_c15_research_probe.sh /absolute/new/build}
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
git -C "$repo" archive b7363977c4e95c62917c475b55496c7929f8c64e submission | tar -x -C "$b/src"
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
cmake --build "$b/cpu" -j2 --target topk_c15_group_full topk_c15_group_raw topk_c15_group_baseline topk_c15_group_baseline_raw topk_c15_group_copy >> "$b/logs/cpu-build.log" 2>&1
cd "$b/run"
for kind in raw full baseline baseline_raw copy; do
 "$b/cpu/topk_c15_group_$kind" > "$b/logs/$kind-cpu.log" 2>&1
 grep 'PASS: 25' "$b/logs/$kind-cpu.log"
done
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j2 --target topk_c15_group_full topk_c15_group_raw topk_c15_group_baseline topk_c15_group_baseline_raw topk_c15_group_copy >> "$b/logs/sim-build.log" 2>&1
for rows in 32 64; do
 for kind in full baseline; do
  msprof op simulator --application="$b/sim/topk_c15_group_$kind $rows 1024 64 1" --soc-version=Ascend910B1 --output="$b/profile-$kind-r$rows" --timeout=5 > "$b/logs/$kind-r$rows.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/$kind-r$rows.log"
  grep -E 'veccore0|PASS:' "$b/logs/$kind-r$rows.log"
 done
done
for kind in baseline_raw copy; do
 msprof op simulator --application="$b/sim/topk_c15_group_$kind 64 1024 64 1" --soc-version=Ascend910B1 --output="$b/profile-$kind-r64" --timeout=5 > "$b/logs/$kind-r64.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/$kind-r64.log"
 grep -E 'veccore0|PASS:' "$b/logs/$kind-r64.log"
done
for shape in '17 1024 64 1' '129 1024 64 8'; do
 set -- $shape
 msprof op simulator --application="$b/sim/topk_c15_group_raw $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$b/profile-raw-r$1-c$4" --timeout=5 > "$b/logs/raw-r$1-c$4.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/raw-r$1-c$4.log"
done
