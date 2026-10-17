#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
repo=/mnt/c/Users/34788/Documents/Codex/2026-10-02/ubantu-wsl-cann/work/cann-edit
b=/home/leo/cann-final-throughput-v2
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
cp -r "$repo/submission" "$b/src/"
cp "$0" "$b/repro.sh"
exec > "$b/driver.log" 2>&1
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
for t in topk_fp16_large_production_probe topk_fp32_large_production_probe topk_c14_block_production_probe; do
 cmake --build "$b/cpu" -j1 --target "$t" >> "$b/logs/cpu-build.log" 2>&1
 (cd "$b/run"; "$b/cpu/$t") > "$b/logs/$t-cpu.log" 2>&1
 grep 'PASS:' "$b/logs/$t-cpu.log"
done
for r in 30 31 32 55 56 57 63 64 65 113; do
 for item in 'half 256 64' 'float 97 32' 'c14 512 10'; do
  set -- $item
  case $1 in half) t=topk_fp16_large_production_probe;;float) t=topk_fp32_large_production_probe;;c14) t=topk_c14_block_production_probe;;esac
  (cd "$b/run"; "$b/cpu/$t" "$r" "$2" "$3" 1) > "$b/logs/extra-$1-r$r.log" 2>&1
  grep -q 'PASS:' "$b/logs/extra-$1-r$r.log"
 done
done
echo CPU_COMPLETE
cmake -S "$b/src/submission" -B "$b/device" -DNPU_ARCH=dav-2201 > "$b/logs/device-build.log" 2>&1
cmake --build "$b/device" -j1 >> "$b/logs/device-build.log" 2>&1
echo DEVICE_COMPLETE
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
for t in topk_fp16_large_production_probe topk_fp32_large_production_probe topk_c14_block_production_probe; do
 cmake --build "$b/sim" -j1 --target "$t" >> "$b/logs/sim-build.log" 2>&1
done
echo SIM_BUILD_COMPLETE
