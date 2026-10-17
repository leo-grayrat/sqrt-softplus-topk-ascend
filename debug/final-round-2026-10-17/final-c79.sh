#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
repo=/mnt/c/Users/34788/Documents/Codex/2026-10-02/ubantu-wsl-cann/work/cann-edit
while [ ! -f /home/leo/cann-final-c14-28/logs/c14-r128-c1-sim.log ]; do sleep 2; done
b=/home/leo/cann-final-c79
mkdir -p "$b/src/debug" "$b/logs" "$b/run"
cp -r "$repo/submission" "$b/src/"
cp -r "$repo/debug/cpu_topk_probe" "$b/src/debug/"
cp "$0" "$b/repro.sh"
exec > "$b/driver.log" 2>&1
t=topk_fp32_scratch_production_probe
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/cpu" -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/cpu-build.log" 2>&1
cmake --build "$b/cpu" -j1 --target "$t" >> "$b/logs/cpu-build.log" 2>&1
(cd "$b/run"; "$b/cpu/$t") > "$b/logs/cpu.log" 2>&1
grep 'PASS:' "$b/logs/cpu.log"
for shape in '31 150 6 1' '32 160 7 1' '33 150 6 1' '65 160 7 1' '157 150 6 1' '3072 160 7 32' '5000 150 6 32'; do
 set -- $shape
 (cd "$b/run"; "$b/cpu/$t" "$1" "$2" "$3" "$4") > "$b/logs/extra-r$1-e$2.log" 2>&1
 grep -q PASS "$b/logs/extra-r$1-e$2.log"
done
echo CPU_COMPLETE
cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/sim" -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/sim-build.log" 2>&1
cmake --build "$b/sim" -j1 --target "$t" >> "$b/logs/sim-build.log" 2>&1
touch "$b/BUILD_COMPLETE"
base=/home/leo/cann-multi-candidate-v2
python3 - "$base" "$repo" <<'PY'
import sys,subprocess
from pathlib import Path
base,repo=map(Path,sys.argv[1:])
old=(base/'src/submission/dense_repeat_override.h').read_text()
main=subprocess.check_output(['git','-C',str(repo),'show','b8bebe:submission/dense_repeat_override.h']).decode()
for token,end in [('class SqrtSoftplusVendorPerfTopKKernel','\n};'),('__global__ __vector__ void sqrt_softplus_prefix_merge_top_k_dispatch_custom','#define sqrt_softplus_prefix_merge_top_k_custom')]:
 def section(s):
  a=s.index(token);b=s.index(end,a);return s[a:b]
 assert section(old)==section(main),token
assert (base/'src/debug/cpu_topk_probe/topk_fp32_scratch_production_probe.asc').read_bytes()==(repo/'debug/cpu_topk_probe/topk_fp32_scratch_production_probe.asc').read_bytes()
print('REUSED_BASELINE_C79_SOURCE_AND_PROBE_MATCH_MAIN')
PY
profile() {
 local root=$1 rows=$2 experts=$3 k=$4 cores=$5 key=final-r$2-e$3-k$4-c$5
 mkdir -p "$root/run-$key"
 (cd "$root/run-$key"; msprof op simulator --application="$root/sim/$t $rows $experts $k $cores" --soc-version=Ascend910B1 --output="$root/profile-$key" --timeout=6) > "$root/logs/$key-sim.log" 2>&1
 grep -q 'PASS: 1' "$root/logs/$key-sim.log"
 echo "$root $key PASS"
}
pids=()
for shape in '96 160 7 1' '96 150 6 1' '3072 160 7 32' '5000 150 6 32'; do
 set -- $shape
 profile "$base" "$1" "$2" "$3" "$4" & pids+=($!)
 profile "$b" "$1" "$2" "$3" "$4" & pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo MODELS_COMPLETE
