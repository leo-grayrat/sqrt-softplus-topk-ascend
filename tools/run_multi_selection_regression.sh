#!/usr/bin/env bash
set -eo pipefail
mode=${1:?baseline or candidate}
b=${2:?existing frozen build}
test -f "$b/BUILD_COMPLETE"
test ! -e "$b/regression-driver.log"
(
for rows in 9 19 20; do
 mkdir -p "$b/run-reg-cpu-r$rows"
 (cd "$b/run-reg-cpu-r$rows"; "$b/cpu/topk_c15_group_production_probe" "$rows" 1024 64 1) > "$b/logs/c15-extra-r$rows-cpu.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/c15-extra-r$rows-cpu.log"
done
profile() {
 local target=$1 rows=$2 experts=$3 k=$4 cores=$5
 local key=r$rows-e$experts-k$k-c$cores
 mkdir -p "$b/run-$key"
 (cd "$b/run-$key"; msprof op simulator --application="$b/sim/$target $rows $experts $k $cores" --soc-version=Ascend910B1 --output="$b/profile-$key" --timeout=5) > "$b/logs/sim-$key.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/sim-$key.log"
}
profile topk_fp32_scratch_production_probe 1 150 6 1
profile topk_fp32_scratch_production_probe 1 160 7 1
profile topk_c15_group_production_probe 1 1024 64 1
if [ "$mode" = candidate ]; then
 profile topk_fp32_scratch_production_probe 49 129 6 8
 profile topk_fp32_scratch_production_probe 49 159 7 8
 profile topk_fp32_scratch_production_probe 129 210 8 8
 profile topk_fp32_scratch_production_probe 257 888 8 8
 profile topk_c15_group_production_probe 17 1023 64 8
 cmake -S "$b/src/submission" -B "$b/device" -DNPU_ARCH=dav-2201 > "$b/logs/device-build.log" 2>&1
 cmake --build "$b/device" -j1 >> "$b/logs/device-build.log" 2>&1
fi
echo COMPLETE
) > "$b/regression-driver.log" 2>&1
