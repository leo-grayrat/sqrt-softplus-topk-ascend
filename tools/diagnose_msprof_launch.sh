#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH 为空；请先 source CANN set_env.sh。" >&2
    exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
BIN="${1:-/tmp/sqrt-softplus-topk-ascend-profile-${USER:-user}/baseline/debug/topk_profile_one/build_sim_profile/topk_profile_one}"
SIM_SOC="${2:-Ascend920A}"
ROWS="${3:-1}"
EXPERTS="${4:-1024}"
K="${5:-8}"
SIM_LIB="$ASCEND_HOME_PATH/tools/simulator/dav_2201/lib"
OUT="$ROOT/profile-output/${SIM_SOC}_r${ROWS}_e${EXPERTS}_k${K}/msprof-debug"
LOG="$OUT/msprof-debug.log"

if [[ ! -x "$BIN" ]]; then
    echo "找不到可执行文件: $BIN" >&2
    exit 1
fi
if [[ ! -d "$SIM_LIB" ]]; then
    echo "找不到 simulator lib: $SIM_LIB" >&2
    exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"

echo "binary:    $BIN"
echo "simulator: $SIM_SOC"
echo "log:       $LOG"
echo

echo "先做一次直接运行 sanity check："
set +e
"$BIN" "$ROWS" "$EXPERTS" "$K" >"$OUT/direct-run.log" 2>&1
DIRECT_RC=$?
set -e
if [[ $DIRECT_RC -eq 0 ]]; then
    echo "[OK] direct run exit=0"
else
    echo "[FAIL] direct run exit=$DIRECT_RC；先不要分析 msprof。" >&2
    tail -n 80 "$OUT/direct-run.log"
    exit "$DIRECT_RC"
fi

echo
echo "运行 msprof，并把 CANN 日志级别提高到 DEBUG；完整输出只写文件。"
set +e
ASCEND_SLOG_PRINT_TO_STDOUT=1 \
ASCEND_GLOBAL_LOG_LEVEL=0 \
LD_LIBRARY_PATH="$SIM_LIB:${LD_LIBRARY_PATH:-}" \
msprof op simulator \
    --soc-version="$SIM_SOC" \
    --output="$OUT/profile" \
    "$BIN" "$ROWS" "$EXPERTS" "$K" \
    >"$LOG" 2>&1
MSPROF_RC=$?
set -e

echo "msprof exit=$MSPROF_RC"
echo

echo "===== 关键日志 ====="
MATCH_RE='AscendCFunctionGetMetaInfo|rtFunctionGetMetaInfo|GetProfkType|kernel[ _-]*type|kType|cross[ _-]*core|core[ _-]*ratio|profil|Launch kernel failure|launch.*failed|ERROR|WARN'
if ! grep -Eia -C 2 "$MATCH_RE" "$LOG" | tail -n 240; then
    echo "没有匹配到 metadata/profiling 关键字，显示日志末尾 100 行："
    tail -n 100 "$LOG"
fi

echo
echo "===== profiling 产物 ====="
find "$OUT/profile" -type f \( -name '*_instr_exe.csv' -o -name '*_code_exe.csv' -o -name 'trace.json' \) -print 2>/dev/null || true

echo
echo "完整日志保存在：$LOG"
