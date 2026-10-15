#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
b=/home/leo/cann-fp32-k8-delivery-lf-v1
repo=/mnt/c/Users/34788/Documents/Codex/2026-10-02/ubantu-wsl-cann/work/cann-edit
test ! -e "$b/src"
mkdir -p "$b/logs" "$b/run"
python3 - "$repo" "$b" <<'PY'
from pathlib import Path
import subprocess,sys
r=Path(sys.argv[1]); b=Path(sys.argv[2])
files=subprocess.check_output(['git','-C',str(r),'ls-files','debug/cpu_topk_probe','submission']).decode().splitlines()
for rel in files:
    p=b/'src'/rel; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(subprocess.check_output(['git','-C',str(r),'show',':'+rel]))
assert b'\r\n' not in (b/'src/submission/dense_repeat_override.h').read_bytes()
print('Frozen staged files:',len(files))
PY
for run in cpu sim; do
 cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/$run" -DCMAKE_ASC_RUN_MODE="$run" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/$run-build.log" 2>&1
 cmake --build "$b/$run" -j2 --target topk_fp32_k8_block_production_probe >> "$b/logs/$run-build.log" 2>&1
done
cd "$b/run"
"$b/cpu/topk_fp32_k8_block_production_probe" > "$b/logs/cpu.log" 2>&1
grep 'PASS: 49' "$b/logs/cpu.log"
msprof op simulator --application="$b/sim/topk_fp32_k8_block_production_probe 129 888 8 8" --soc-version=Ascend910B1 --output="$b/profile-r129-e888-k8-c8" --timeout=5 > "$b/logs/sim-r129-e888-k8-c8.log" 2>&1
grep 'PASS: 1' "$b/logs/sim-r129-e888-k8-c8.log"
cmake -S "$b/src/submission" -B "$b/device" -DNPU_ARCH=dav-2201 > "$b/logs/device-build.log" 2>&1
cmake --build "$b/device" -j1 >> "$b/logs/device-build.log" 2>&1
printf 'COMPLETE\n'
