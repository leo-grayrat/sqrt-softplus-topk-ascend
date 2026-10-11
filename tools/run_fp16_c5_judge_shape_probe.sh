#!/usr/bin/env bash
# Additional whole-case shape check using a previously frozen build.
# Source CANN set_env.sh first. This is simulator evidence, not a Judge run.
set -eo pipefail
build=${1:?Usage: run_fp16_c5_judge_shape_probe.sh /absolute/build}
case "$build" in /*) ;; *) exit 2 ;; esac
target=topk_fp16_compact_production_probe
mkdir -p "$build/run-judge-shape"
cd "$build/run-judge-shape"
"$build/cpu/$target" 512 200 16 32 > "$build/logs/cpu-r512-e200-k16-c32.log" 2>&1
grep 'PASS: 1 fp16 production compact cases' "$build/logs/cpu-r512-e200-k16-c32.log"
msprof op simulator --application="$build/sim/$target 512 200 16 32" \
    --soc-version=Ascend910B1 --output="$build/profile-r512-e200-k16-c32" --timeout=5 \
    > "$build/logs/sim-r512-e200-k16-c32.log" 2>&1
grep -E 'veccore0|PASS:' "$build/logs/sim-r512-e200-k16-c32.log"
grep -q 'PASS: 1 fp16 production compact cases' "$build/logs/sim-r512-e200-k16-c32.log"
