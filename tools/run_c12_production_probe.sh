#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?Usage: run_c12_production_probe.sh baseline|candidate /absolute/new/build}
build=${2:?Missing build path}
case "$mode" in baseline|candidate) ;; *) exit 2 ;; esac
case "$build" in /*) ;; *) exit 2 ;; esac
test ! -e "$build/src"
mkdir -p "$build/src/debug" "$build/logs" "$build/run"
cp "$0" "$build/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$build/src/debug/"
if [ "$mode" = baseline ]; then
 git -C "$repo" archive deb10330f79908c0f8929d3a5cf236e0fd128eed submission | tar -x -C "$build/src"
else
 cp -r "$repo/submission" "$build/src/"
fi
target=topk_c12_block_production_probe
for run in cpu sim; do
 cmake -S "$build/src/debug/cpu_topk_probe" -B "$build/$run" -DCMAKE_ASC_RUN_MODE="$run" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$build/logs/$run-build.log" 2>&1
 cmake --build "$build/$run" -j2 --target "$target" >> "$build/logs/$run-build.log" 2>&1
done
cd "$build/run"
"$build/cpu/$target" > "$build/logs/cpu.log" 2>&1
grep 'PASS: 43' "$build/logs/cpu.log"
for shape in '96 256 3 1' '192 256 3 1' '1 256 3 1' '65 256 3 1' '257 256 3 8' '32768 256 3 32'; do
 set -- $shape
 timeout=5
 if [ "$1" = 32768 ]; then timeout=10; fi
 msprof op simulator --application="$build/sim/$target $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-k$3-c$4" --timeout="$timeout" > "$build/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
 grep -q 'PASS: 1' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
 grep -E 'veccore0|PASS:' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
done
if [ "$mode" = candidate ]; then
 for shape in '33 255 3 8' '64 256 4 1' '17 256 2 8' '33 200 16 8' '33 512 10 8' '33 256 64 8'; do
  set -- $shape
  msprof op simulator --application="$build/sim/$target $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$build/profile-r$1-e$2-k$3-c$4" --timeout=5 > "$build/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
  grep -q 'PASS: 1' "$build/logs/sim-r$1-e$2-k$3-c$4.log"
 done
 cmake -S "$build/src/submission" -B "$build/device" -DNPU_ARCH=dav-2201 > "$build/logs/device-build.log" 2>&1
 cmake --build "$build/device" -j1 >> "$build/logs/device-build.log" 2>&1
fi
