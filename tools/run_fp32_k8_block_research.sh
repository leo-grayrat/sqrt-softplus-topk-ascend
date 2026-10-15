#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
b=${1:?Usage: run_fp32_k8_block_research.sh /absolute/new/build}
case "$b" in /*) ;; *) exit 2 ;; esac
test ! -e "$b/src"
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp "$0" "$b/repro-snapshot.sh"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
git -C "$repo" archive c2482715d319e1b93065c239fa1bb9b997a7916a submission | tar -x -C "$b/src"
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
cmake --build "$b/cpu" -j2 --target topk_fp32_k8_block_full topk_fp32_k8_block_raw topk_fp32_k8_block_baseline >> "$b/logs/cpu-build.log" 2>&1
cd "$b/run"
for kind in raw full baseline; do
 "$b/cpu/topk_fp32_k8_block_$kind" > "$b/logs/$kind-cpu.log" 2>&1
 grep 'PASS: 38' "$b/logs/$kind-cpu.log"
done
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j2 --target topk_fp32_k8_block_full topk_fp32_k8_block_raw topk_fp32_k8_block_baseline >> "$b/logs/sim-build.log" 2>&1
for shape in '32 888' '64 888' '64 210' '128 210'; do
 set -- $shape
 for kind in full baseline; do
  msprof op simulator --application="$b/sim/topk_fp32_k8_block_$kind $1 $2 8 1" --soc-version=Ascend910B1 --output="$b/profile-$kind-r$1-e$2" --timeout=5 > "$b/logs/$kind-r$1-e$2.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/$kind-r$1-e$2.log"
  grep -E 'veccore0|PASS:' "$b/logs/$kind-r$1-e$2.log"
 done
done
for shape in '17 888 1' '129 210 8'; do
 set -- $shape
 msprof op simulator --application="$b/sim/topk_fp32_k8_block_raw $1 $2 8 $3" --soc-version=Ascend910B1 --output="$b/profile-raw-r$1-e$2-c$3" --timeout=5 > "$b/logs/raw-r$1-e$2-c$3.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/raw-r$1-e$2-c$3.log"
done
