#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?baseline or candidate}
b=${2:?new absolute build directory}
case "$mode" in baseline|candidate) ;; *) exit 2 ;; esac
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
if [ "$mode" = baseline ]; then
 git -C "$repo" archive 92189c1e0d2339ed6113785ac8810007d750cb50 submission | tar -x -C "$b/src"
else
 cp -r "$repo/submission" "$b/src/"
fi
target=topk_c14_block_production_probe
for run in cpu sim; do
 cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/$run" -DCMAKE_ASC_RUN_MODE="$run" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/$run-build.log" 2>&1
 cmake --build "$b/$run" -j2 --target "$target" topk_c13_selection_production_probe >> "$b/logs/$run-build.log" 2>&1
done
cd "$b/run"
"$b/cpu/$target" > "$b/logs/cpu.log" 2>&1
grep 'PASS: 31' "$b/logs/cpu.log"
"$b/cpu/topk_c13_selection_production_probe" > "$b/logs/float-cpu.log" 2>&1
grep 'PASS: 49' "$b/logs/float-cpu.log"
for rows in 23 24 25 47 48 49; do
 "$b/cpu/$target" "$rows" 512 10 1 > "$b/logs/extra-cpu-r$rows.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/extra-cpu-r$rows.log"
done
for shape in '64 512 10 1' '128 512 10 1' '1 512 10 1' '33 512 10 1' '249 512 10 8' '65536 512 10 32' '33 511 10 1' '65 256 3 8' '33 256 64 8'; do
 set -- $shape
 msprof op simulator --application="$b/sim/$target $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$b/profile-r$1-e$2-k$3-c$4" --timeout=20 > "$b/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
 grep -E 'veccore0|PASS:' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
done
for shape in '257 888 8 8' '129 210 8 8' '65 1024 64 8'; do
 set -- $shape
 msprof op simulator --application="$b/sim/topk_c13_selection_production_probe $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$b/profile-r$1-e$2-k$3-c$4" --timeout=5 > "$b/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
done
if [ "$mode" = candidate ]; then
 cmake -S "$b/src/submission" -B "$b/device" -DNPU_ARCH=dav-2201 > "$b/logs/device-build.log" 2>&1
 cmake --build "$b/device" -j1 >> "$b/logs/device-build.log" 2>&1
fi
echo COMPLETE
