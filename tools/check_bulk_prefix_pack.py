#!/usr/bin/env python3
from pathlib import Path
import re
import sys

HEADER = Path(__file__).resolve().parents[1] / "submission" / "dense_repeat_override.h"
text = HEADER.read_text(encoding="utf-8")

signature = "__aicore__ inline void PackPrefixes("
start = text.find(signature)
if start < 0:
    raise SystemExit("FAIL: PackPrefixes() not found")

brace = text.find("{", start)
if brace < 0:
    raise SystemExit("FAIL: PackPrefixes() body not found")

depth = 0
end = None
for i in range(brace, len(text)):
    if text[i] == "{":
        depth += 1
    elif text[i] == "}":
        depth -= 1
        if depth == 0:
            end = i + 1
            break
if end is None:
    raise SystemExit("FAIL: unterminated PackPrefixes() body")

body = text[brace:end]

checks = [
    ("AscendC::DataCopyParams", "missing DataCopyParams"),
    ("blockCount", "missing blockCount"),
    ("blockLen", "missing blockLen"),
    ("srcStride", "missing srcStride"),
    ("dstStride", "missing dstStride"),
]
for needle, message in checks:
    if needle not in body:
        raise SystemExit(f"FAIL: {message}")

if re.search(r"\bfor\s*\(", body):
    raise SystemExit("FAIL: PackPrefixes() still loops over queues")

if body.count("AscendC::DataCopy(") != 1:
    raise SystemExit("FAIL: PackPrefixes() must issue exactly one DataCopy")

if body.count("AscendC::PipeBarrier<PIPE_V>()") != 1:
    raise SystemExit("FAIL: PackPrefixes() must keep exactly one PIPE_V barrier")

print("PASS: PackPrefixes uses one strided DataCopy and one PIPE_V barrier")
