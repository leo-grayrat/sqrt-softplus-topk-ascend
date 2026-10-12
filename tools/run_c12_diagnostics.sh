#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${1:?Usage: run_c12_diagnostics.sh /absolute/new/build}
case "$build" in /*) ;; *) exit 2 ;; esac
for kind in raw copy; do
 b="$build/$kind"
 test ! -e "$b/src"
 mkdir -p "$b/logs" "$b/run"
 python3 "$repo/tools/prepare_c12_diagnostics.py" "$b/src" "$kind"
 for mode in cpu sim; do
  cmake -S "$b/src/debug/cpu_topk_probe" -B "$b/$mode" -DCMAKE_ASC_RUN_MODE="$mode" -DCMAKE_ASC_ARCHITECTURES=dav-2201 > "$b/logs/$mode-build.log" 2>&1
  cmake --build "$b/$mode" -j2 --target topk_c12_diagnostic >> "$b/logs/$mode-build.log" 2>&1
 done
 cd "$b/run"
 "$b/cpu/topk_c12_diagnostic" > "$b/logs/cpu.log" 2>&1
 grep 'PASS: 33' "$b/logs/cpu.log"
 for rows in 62 124; do
  msprof op simulator --application="$b/sim/topk_c12_diagnostic $rows 256 3 1" --soc-version=Ascend910B1 --output="$b/profile-r$rows" --timeout=5 > "$b/logs/sim-r$rows.log" 2>&1
  grep -q 'PASS: 1' "$b/logs/sim-r$rows.log"
 done
done
