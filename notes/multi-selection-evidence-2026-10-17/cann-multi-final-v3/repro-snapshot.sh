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
target=topk_multi_half_production_probe
targets="$target"
if [ "$mode" = candidate ]; then targets="$targets topk_fp32_scratch_production_probe"; fi
for run in cpu sim; do
 cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/$run" -DCMAKE_ASC_RUN_MODE="$run" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/$run-build.log" 2>&1
 cmake --build "$b/$run" -j1 --target $targets >> "$b/logs/$run-build.log" 2>&1
 if [ "$run" = cpu ]; then
  mkdir -p "$b/run-cpu"
  (cd "$b/run-cpu"; "$b/cpu/$target") > "$b/logs/half-cpu.log" 2>&1
  grep 'PASS: 29' "$b/logs/half-cpu.log"
  if [ "$mode" = candidate ]; then
   (cd "$b/run-cpu"; "$b/cpu/topk_fp32_scratch_production_probe") > "$b/logs/float-cpu.log" 2>&1
   grep 'PASS: 87' "$b/logs/float-cpu.log"
   (cd "$b/run-cpu"; "$b/cpu/topk_fp32_scratch_production_probe" 23 1024 64 1) > "$b/logs/float-c15-cpu.log" 2>&1
   grep -q 'PASS: 1' "$b/logs/float-c15-cpu.log"
  fi
 fi
done
touch "$b/BUILD_COMPLETE"
profile() {
 local exe=$1 rows=$2 experts=$3 k=$4 cores=$5
 local key=r$rows-e$experts-k$k-c$cores
 mkdir -p "$b/run-$key"
 (cd "$b/run-$key"; msprof op simulator --application="$b/sim/$exe $rows $experts $k $cores" --soc-version=Ascend910B1 --output="$b/profile-$key" --timeout=8) > "$b/logs/sim-$key.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/sim-$key.log"
}
for rows in 96 192; do profile "$target" "$rows" 128 16 1; done
profile "$target" 4096 128 16 32
profile "$target" 1 128 16 1
profile "$target" 49 128 16 8
if [ "$mode" = candidate ]; then
 profile "$target" 33 127 16 8
 profile "$target" 33 128 17 8
 profile "$target" 65 512 10 8
 profile topk_fp32_scratch_production_probe 96 150 6 1
 profile topk_fp32_scratch_production_probe 96 160 7 1
 profile topk_fp32_scratch_production_probe 64 1024 64 1
 cmake -S "$b/src/submission" -B "$b/device" -DNPU_ARCH=dav-2201 > "$b/logs/device-build.log" 2>&1
 cmake --build "$b/device" -j1 >> "$b/logs/device-build.log" 2>&1
fi
echo COMPLETE
