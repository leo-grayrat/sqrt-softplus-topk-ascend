#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
b=/home/leo/cann-final-throughput-v2
repo=/mnt/c/Users/34788/Documents/Codex/2026-10-02/ubantu-wsl-cann/work/cann-edit
while ! grep -q SIM_BUILD_COMPLETE "$b/driver.log"; do sleep 3; done
base=/home/leo/cann-final-throughput-baseline
mkdir -p "$base/src/debug" "$base/logs"
cp -r "$b/src/debug/cpu_topk_probe" "$base/src/debug/"
git -C /home/leo/sqrt-softplus-topk-ascend archive b8bebe9307de2adb503a92bb470ae18b789a9a13 submission | tar -x -C "$base/src"
exec > "$b/models-driver.log" 2>&1
cmake -S "$base/src/debug/cpu_topk_probe" -B "$base/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$base/logs/sim-build.log" 2>&1
for t in topk_fp16_large_production_probe topk_fp32_large_production_probe; do
 cmake --build "$base/sim" -j1 --target "$t" >> "$base/logs/sim-build.log" 2>&1
done
profile() {
 local root=$1 kind=$2 rows=$3 experts=$4 k=$5 cores=$6
 local t=topk_fp32_large_production_probe
 if [ "$kind" = half ]; then t=topk_fp16_large_production_probe; fi
 local key=$kind-r$rows-e$experts-k$k-c$cores
 mkdir -p "$root/run-$key"
 (cd "$root/run-$key"; msprof op simulator --application="$root/sim/$t $rows $experts $k $cores" --soc-version=Ascend910B1 --output="$root/profile-$key" --timeout=8) > "$root/logs/$key-sim.log" 2>&1
 grep -q 'PASS: 1' "$root/logs/$key-sim.log"
 echo "$root $key PASS"
}
pids=()
for shape in 'half 128 256 64 1' 'float 128 97 32 1' 'half 4096 256 64 32' 'float 4096 97 32 32'; do
 set -- $shape
 profile "$base" "$1" "$2" "$3" "$4" "$5" & pids+=($!)
 profile "$b" "$1" "$2" "$3" "$4" "$5" & pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo MODELS_COMPLETE
