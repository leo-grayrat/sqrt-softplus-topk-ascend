#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?Usage: run_fp32_k8_block_production.sh baseline|candidate /absolute/new/build}
b=${2:?Missing build path}
case "$mode" in baseline|candidate) ;; *) exit 2 ;; esac
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
if [ "$mode" = baseline ]; then
 git -C "$repo" archive c2482715d319e1b93065c239fa1bb9b997a7916a submission | tar -x -C "$b/src"
else
 cp -r "$repo/submission" "$b/src/"
fi
target=topk_fp32_k8_block_production_probe
for run in cpu sim; do
 cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/$run" -DCMAKE_ASC_RUN_MODE="$run" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/$run-build.log" 2>&1
 cmake --build "$b/$run" -j2 --target "$target" >> "$b/logs/$run-build.log" 2>&1
done
cd "$b/run"
"$b/cpu/$target" > "$b/logs/cpu.log" 2>&1
grep 'PASS: 49' "$b/logs/cpu.log"
for shape in '32 888 8 1' '64 888 8 1' '1 888 8 1' '7777 888 8 32' \
             '64 210 8 1' '128 210 8 1' '1 210 8 1' '32999 210 8 32'; do
 set -- $shape
 msprof op simulator --application="$b/sim/$target $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$b/profile-r$1-e$2-k$3-c$4" --timeout=15 > "$b/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
 grep -E 'veccore0|PASS:' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
done
if [ "$mode" = candidate ]; then
 for shape in '33 887 8 8' '33 211 8 8' '33 888 7 8' '33 210 7 8' '9 888 9 8' '65 1024 64 8'; do
  set -- $shape
  msprof op simulator --application="$b/sim/$target $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$b/profile-r$1-e$2-k$3-c$4" --timeout=5 > "$b/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
 done
 for run in cpu sim; do
  cmake --build "$b/$run" -j2 --target topk_c14_block_production_probe > "$b/logs/half-$run-build.log" 2>&1
 done
 "$b/cpu/topk_c14_block_production_probe" > "$b/logs/half-cpu.log" 2>&1
 grep 'PASS: 31' "$b/logs/half-cpu.log"
 for shape in '65 256 3 8' '65 512 10 8'; do
  set -- $shape
  msprof op simulator --application="$b/sim/topk_c14_block_production_probe $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$b/profile-half-r$1-e$2-k$3-c$4" --timeout=5 > "$b/logs/half-sim-r$1-e$2-k$3-c$4.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/half-sim-r$1-e$2-k$3-c$4.log"
 done
 cmake -S "$b/src/submission" -B "$b/device" -DNPU_ARCH=dav-2201 > "$b/logs/device-build.log" 2>&1
 cmake --build "$b/device" -j1 >> "$b/logs/device-build.log" 2>&1
fi
