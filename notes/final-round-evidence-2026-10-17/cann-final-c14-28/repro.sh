#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
prev=/home/leo/cann-final-throughput-v2
# The first model log is created only after all other compiler jobs finish.
while [ ! -f "$prev/logs/half-r128-e256-k64-c1-sim.log" ]; do sleep 3; done
b=/home/leo/cann-final-c14-28
mkdir -p "$b/src" "$b/logs" "$b/run"
cp -r "$prev/src/"* "$b/src/"
cp "$0" "$b/repro.sh"
python3 - "$b/src/submission/dense_repeat_override.h" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text();a=s.index('class SqrtSoftplusBlockMaxMediumTopKHalfKernel');b=s.index('class SqrtSoftplusBlockMaxTopKHalfKernel',a)
c=s[a:b].replace('batch_ = perCore < 24 ? perCore : 24;', 'const int32_t batches = (perCore + 27) / 28;\n        batch_ = (perCore + batches - 1) / batches;')
p.write_text(s[:a]+c+s[b:])
PY
exec > "$b/driver.log" 2>&1
t=topk_c14_block_production_probe
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
cmake --build "$b/cpu" -j1 --target "$t" >> "$b/logs/cpu-build.log" 2>&1
(cd "$b/run"; "$b/cpu/$t") > "$b/logs/cpu.log" 2>&1
grep 'PASS:' "$b/logs/cpu.log"
for rows in 27 28 29 55 56 57; do
 (cd "$b/run"; "$b/cpu/$t" "$rows" 512 10 1) > "$b/logs/extra-r$rows.log" 2>&1
 grep -q PASS "$b/logs/extra-r$rows.log"
done
echo CPU_COMPLETE
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j1 --target "$t" >> "$b/logs/sim-build.log" 2>&1
profile() {
 local root=$1 rows=$2 cores=$3
 local key=c14-r$rows-c$cores
 mkdir -p "$root/run-$key"
 (cd "$root/run-$key"; msprof op simulator --application="$root/sim/$t $rows 512 10 $cores" --soc-version=Ascend910B1 --output="$root/profile-$key" --timeout=7) > "$root/logs/$key-sim.log" 2>&1
 grep -q 'PASS: 1' "$root/logs/$key-sim.log"
 echo "$root $key PASS"
}
pids=()
for shape in '128 1' '4096 32'; do
 set -- $shape
 profile "$prev" "$1" "$2" & pids+=($!)
 profile "$b" "$1" "$2" & pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo MODELS_COMPLETE
