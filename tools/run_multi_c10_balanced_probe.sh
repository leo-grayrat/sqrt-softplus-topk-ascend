#!/usr/bin/env bash
# Recheck the changed half path only; frozen FP32 classes are unchanged.
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?new absolute build directory}
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
cp -r "$repo/submission" "$b/src/"
target=topk_multi_half_production_probe
for run in cpu sim; do
 cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/$run" -DCMAKE_ASC_RUN_MODE="$run" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/$run-build.log" 2>&1
 cmake --build "$b/$run" -j1 --target "$target" >> "$b/logs/$run-build.log" 2>&1
 if [ "$run" = cpu ]; then
  mkdir -p "$b/run-cpu"
  (cd "$b/run-cpu"; "$b/cpu/$target") > "$b/logs/half-cpu.log" 2>&1
  grep 'PASS: 29' "$b/logs/half-cpu.log"
 fi
done
profile() {
 local rows=$1 experts=$2 k=$3 cores=$4
 local key=r$rows-e$experts-k$k-c$cores
 mkdir -p "$b/run-$key"
 (cd "$b/run-$key"; msprof op simulator --application="$b/sim/$target $rows $experts $k $cores" --soc-version=Ascend910B1 --output="$b/profile-$key" --timeout=8) > "$b/logs/sim-$key.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/sim-$key.log"
}
for rows in 96 192; do profile "$rows" 128 16 1; done
profile 4096 128 16 32
profile 1 128 16 1
profile 49 128 16 8
profile 33 127 16 8
profile 33 128 17 8
profile 65 512 10 8
cmake -S "$b/src/submission" -B "$b/device" -DNPU_ARCH=dav-2201 > "$b/logs/device-build.log" 2>&1
cmake --build "$b/device" -j1 >> "$b/logs/device-build.log" 2>&1
echo COMPLETE
