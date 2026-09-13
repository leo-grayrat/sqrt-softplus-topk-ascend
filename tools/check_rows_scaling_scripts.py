#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
profile = (ROOT / "tools" / "profile_rows_scaling.sh").read_text(encoding="utf-8")
summary = (ROOT / "tools" / "summarize_rows_scaling.py").read_text(encoding="utf-8")

profile_checks = [
    ("ROWS_LIST=(1 2 4 8)", "rows sweep must be 1/2/4/8"),
    ("BLOCK_NUM=1", "rows sweep must force blockNum=1"),
    ('"$BUILD/topk_profile_one" "$rows" "$EXPERTS" "$K" "$BLOCK_NUM"',
     "profile driver must receive blockNum=1"),
    ("origin/main", "rows scaling must profile the production dense baseline"),
]
for needle, message in profile_checks:
    if needle not in profile:
        raise SystemExit(f"FAIL: {message}")

summary_checks = [
    ("ROWS = (1, 2, 4, 8)", "summarizer rows mismatch"),
    ("fit_linear", "summarizer must fit fixed + per-row cost"),
    ("SCALAR", "summarizer must report scalar cost"),
    ("per_row_share", "summarizer must estimate repeated-row share"),
]
for needle, message in summary_checks:
    if needle not in summary:
        raise SystemExit(f"FAIL: {message}")

print("PASS: rows scaling scripts keep the single-block experiment invariant")
