#!/usr/bin/env bash
set -eo pipefail
source /home/leo/Ascend/cann-9.0.0/set_env.sh
b=/home/leo/cann-multi-final-v4
test -f "$b/cpu/topk_multi_half_production_probe"
grep -q 'PASS: 29' "$b/logs/half-cpu.log"
test ! -e "$b/extra-driver.log"
(
for shape in '43 1' '48 1' '128 1' '16384 32'; do
 set -- $shape
 mkdir -p "$b/run-extra-r$1-c$2"
 (cd "$b/run-extra-r$1-c$2"; "$b/cpu/topk_multi_half_production_probe" "$1" 128 16 "$2") > "$b/logs/extra-half-r$1-c$2-cpu.log" 2>&1
 grep -q 'PASS: 1' "$b/logs/extra-half-r$1-c$2-cpu.log"
done
echo COMPLETE
) > "$b/extra-driver.log" 2>&1
