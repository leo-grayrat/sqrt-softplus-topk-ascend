#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?new absolute build directory}
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
cp -r "$repo/submission" "$b/src/"
target=topk_c15_group_production_probe
(
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
cmake --build "$b/cpu" -j1 --target "$target" >> "$b/logs/cpu-build.log" 2>&1
mkdir -p "$b/run-cpu"
(cd "$b/run-cpu"; "$b/cpu/$target") > "$b/logs/default-cpu.log" 2>&1
grep 'PASS: 51' "$b/logs/default-cpu.log"
for mode in 1 2 3 4 5 6 7 8; do
 (cd "$b/run-cpu"; FILTER_INPUT_MODE=$mode "$b/cpu/$target" 23 1024 64 1) > "$b/logs/extra-mode$mode-cpu.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/extra-mode$mode-cpu.log"
done
(cd "$b/run-cpu"; FILTER_INPUT_MODE=1 "$b/cpu/$target" 1043 1024 64 32) > "$b/logs/normal-multi-cpu.log" 2>&1
grep -q 'PASS: 1' "$b/logs/normal-multi-cpu.log"
echo CPU_COMPLETE
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j1 --target "$target" >> "$b/logs/sim-build.log" 2>&1

baseline=$b/baseline
mkdir -p "$baseline/src/debug" "$baseline/logs"
cp -r "$b/src/debug/cpu_topk_probe" "$baseline/src/debug/"
git -C "$repo" archive b8bebe9307de2adb503a92bb470ae18b789a9a13 submission | tar -x -C "$baseline/src"
cmake -S "$baseline/src/debug/cpu_topk_probe" -B "$baseline/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$baseline/logs/sim-build.log" 2>&1
cmake --build "$baseline/sim" -j1 --target "$target" >> "$baseline/logs/sim-build.log" 2>&1

profile() {
 local root=$1 mode=$2 rows=$3 cores=$4
 local key=mode$mode-r$rows-c$cores
 mkdir -p "$root/run-$key"
 (cd "$root/run-$key"; FILTER_INPUT_MODE=$mode msprof op simulator --application="$root/sim/$target $rows 1024 64 $cores" --soc-version=Ascend910B1 --output="$root/profile-$key" --timeout=8) > "$root/logs/$key-sim.log" 2>&1
 grep -q 'PASS: 1' "$root/logs/$key-sim.log"
}
# Separate working directories permit independent A/B models in parallel.
for shape in '1 64 1' '1 1043 32' '0 64 1'; do
 set -- $shape
 profile "$baseline" "$1" "$2" "$3" & left=$!
 profile "$b" "$1" "$2" "$3" & right=$!
 wait "$left"; wait "$right"
done
cmake -S "$b/src/submission" -B "$b/device" -DNPU_ARCH=dav-2201 > "$b/logs/device-build.log" 2>&1
cmake --build "$b/device" -j1 >> "$b/logs/device-build.log" 2>&1
echo COMPLETE
) > "$b/driver.log" 2>&1
