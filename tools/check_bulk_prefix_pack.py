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

# Dense repeat 每轮把 queueCount 按 4 路归并并返回 floor(queueCount / 4)。
# 因此只允许那些不会在后续层出现 5/6/7 条队列的 chunkCount。
gate_match = re.search(
    r"const bool useDenseRepeat\s*=\s*(.*?);",
    text,
    re.DOTALL,
)
if not gate_match:
    raise SystemExit("FAIL: useDenseRepeat gate not found")
gate = gate_match.group(1)

required_gate_terms = [
    "chunkCount % 4 == 0",
    "chunkCount <= 16",
    "chunkCount == 32",
]
for term in required_gate_terms:
    if term not in gate:
        raise SystemExit(f"FAIL: dense-repeat gate missing safety term: {term}")

safe_chunks = [
    chunk
    for chunk in range(4, 33)
    if chunk % 4 == 0 and (chunk <= 16 or chunk == 32)
]
if safe_chunks != [4, 8, 12, 16, 32]:
    raise SystemExit(f"FAIL: unexpected dense-repeat safe set: {safe_chunks}")

for unsafe in (20, 24, 28):
    if unsafe in safe_chunks:
        raise SystemExit(f"FAIL: unsafe chunkCount {unsafe} must fall back")

print("PASS: PackPrefixes uses one strided DataCopy and one PIPE_V barrier")
print("PASS: dense-repeat gate allows only chunkCount 4/8/12/16/32")
