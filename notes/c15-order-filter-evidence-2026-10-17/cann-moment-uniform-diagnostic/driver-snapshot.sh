#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
b=/home/leo/cann-moment-uniform-diagnostic
test ! -e "$b"
mkdir -p "$b/logs"
cp "$0" "$b/driver-snapshot.sh"
(
profile() {
 local kind=$1 mode=$2 root=/home/leo/cann-batch-filter-production-v1
 if [ "$kind" = baseline ]; then root=$root/baseline; fi
 local key=$kind-mode$mode
 mkdir -p "$b/run-$key"
 (cd "$b/run-$key"; FILTER_INPUT_MODE=$mode msprof op simulator --application="$root/sim/topk_c15_group_production_probe 64 1024 64 1" --soc-version=Ascend910B1 --output="$b/profile-$key" --timeout=8) > "$b/logs/$key-sim.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/$key-sim.log"
}
profile baseline 2 & left=$!
profile moment 2 & right=$!
wait "$left"; wait "$right"
echo COMPLETE
) > "$b/driver.log" 2>&1
