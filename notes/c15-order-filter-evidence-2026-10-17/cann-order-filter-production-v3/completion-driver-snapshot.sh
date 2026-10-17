#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
b=/home/leo/cann-order-filter-production-v3
test -f "$b/logs/device-eager-build.log"
cp "$0" "$b/completion-driver-snapshot.sh"
(
profile() {
 local root=$1 rows=$2 cores=$3 key=mode2-r$2-c$3
 mkdir -p "$root/run-$key"
 (cd "$root/run-$key"; FILTER_INPUT_MODE=2 msprof op simulator --application="$root/sim/topk_c15_group_production_probe $rows 1024 64 $cores" --soc-version=Ascend910B1 --output="$root/profile-$key" --timeout=12) > "$root/logs/$key-sim.log" 2>&1
 grep -q 'PASS: 1' "$root/logs/$key-sim.log"
 ! grep -q 'The timeout has reached' "$root/logs/$key-sim.log"
}
profile "$b" 128 1 & p1=$!
profile "$b/baseline" 128 1 & p2=$!
profile "$b" 1043 32 & p3=$!
profile "$b/baseline" 1043 32 & p4=$!
wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"
cmake --build "$b/device" -j1 > "$b/logs/device-build.log" 2>&1
echo COMPLETE
) > "$b/completion-driver.log" 2>&1
