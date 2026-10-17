#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?baseline or candidate}
b=${2:?new absolute build directory}
case "$mode" in baseline|candidate) ;; *) exit 2 ;; esac
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
if [ "$mode" = baseline ]; then
 git -C "$repo" archive 3e44aed0f98fab2985ea4397bee1240e14ec2177 submission | tar -x -C "$b/src"
else
 cp -r "$repo/submission" "$b/src/"
fi
for run in cpu sim; do
 cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/$run" -DCMAKE_ASC_RUN_MODE="$run" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/$run-build.log" 2>&1
 cmake --build "$b/$run" -j1 --target topk_fp32_scratch_production_probe topk_c15_group_production_probe >> "$b/logs/$run-build.log" 2>&1
 if [ "$run" = cpu ]; then
  for target in topk_fp32_scratch_production_probe topk_c15_group_production_probe; do
   mkdir -p "$b/run-cpu-$target"
   (cd "$b/run-cpu-$target"; "$b/cpu/$target") > "$b/logs/$target-cpu.log" 2>&1
   grep 'PASS:' "$b/logs/$target-cpu.log"
  done
  for rows in 10 11 12 21 22 23; do
   (cd "$b/run-cpu-topk_c15_group_production_probe"; "$b/cpu/topk_c15_group_production_probe" "$rows" 1024 64 1) > "$b/logs/c15-extra-r$rows-cpu.log" 2>&1
   grep -q 'PASS: 1' "$b/logs/c15-extra-r$rows-cpu.log"
  done
 fi
done
touch "$b/BUILD_COMPLETE"
profile() {
 local target=$1 rows=$2 experts=$3 k=$4 cores=$5
 local key=r$rows-e$experts-k$k-c$cores
 mkdir -p "$b/run-$key"
 (cd "$b/run-$key"; msprof op simulator --application="$b/sim/$target $rows $experts $k $cores" --soc-version=Ascend910B1 --output="$b/profile-$key" --timeout=8) > "$b/logs/sim-$key.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/sim-$key.log"
}
# Independent runs have separate process working directories. No shared dumps.
for rows in 96 192; do
 profile topk_fp32_scratch_production_probe "$rows" 150 6 1 & first=$!
 profile topk_fp32_scratch_production_probe "$rows" 160 7 1 & second=$!
 wait "$first"; wait "$second"
done
for rows in 32 64; do
 profile topk_c15_group_production_probe "$rows" 1024 64 1
done
profile topk_fp32_scratch_production_probe 385 150 6 8 & first=$!
profile topk_fp32_scratch_production_probe 385 160 7 8 & second=$!
wait "$first"; wait "$second"
profile topk_c15_group_production_probe 1043 1024 64 32
echo COMPLETE
