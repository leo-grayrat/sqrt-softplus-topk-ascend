#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
for mode in baseline candidate; do
 b=/home/leo/cann-c14-ten-production-$mode-v1
 test ! -e "$b/resume-driver.log"
 (
 cd "$b/run"
 target=topk_c14_block_production_probe
 for shape in '4096 512 10 32' '33 511 10 1' '65 256 3 8' '33 256 64 8'; do
  set -- $shape
  msprof op simulator --application="$b/sim/$target $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$b/profile-r$1-e$2-k$3-c$4" --timeout=8 > "$b/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
  grep -E 'veccore0|PASS:' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
 done
 for shape in '257 888 8 8' '129 210 8 8' '65 1024 64 8'; do
  set -- $shape
  msprof op simulator --application="$b/sim/topk_c13_selection_production_probe $1 $2 $3 $4" --soc-version=Ascend910B1 --output="$b/profile-r$1-e$2-k$3-c$4" --timeout=5 > "$b/logs/sim-r$1-e$2-k$3-c$4.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/sim-r$1-e$2-k$3-c$4.log"
 done
 if [ "$mode" = candidate ]; then
  cmake --build "$b/device" -j1 > "$b/logs/device-build.log" 2>&1
 fi
 echo 'COMPLETE bounded resume; full 65536-row attempt remains incomplete'
 ) > "$b/resume-driver.log" 2>&1
done
