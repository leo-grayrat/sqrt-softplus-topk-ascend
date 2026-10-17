#!/usr/bin/env python3
"""Host proof of the five-leaf copy schedule; not a CANN/device test.

Every even address contains an original FP32 score's uint32 bit pattern.
Odd addresses contain the unused zero index payload. DataCopy is emulated
in float-slot units using the kernel's exact 32-byte block parameters.
The independent oracle sorts the original FP32 inputs by decoded values
and compares selected score bits exactly, without a numerical tolerance.
"""
import argparse
import json
import math
import random
import struct

SEED = 83079
NEG_INF = 0xFF800000
UNINITIALIZED = None
MODES = ("random", "ties", "all_negative_inf", "tail_winners",
         "first_winners", "one_finite", "cross_leaf_ties", "fp32_edges")


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def value(score_bits):
    return struct.unpack("<f", struct.pack("<I", score_bits))[0]


def order(score_bits):
    # IEEE finite/inf order. Generated rows do not mix positive/negative zero,
    # so bit-level zero ordering does not strengthen the score-tie contract.
    return (~score_bits & 0xFFFFFFFF) if score_bits & 0x80000000 else score_bits ^ 0x80000000


def copy(dst, dst_base, src, src_base, count, block_len, src_stride, dst_stride):
    """Exact DataCopyParams emulation; length/strides count 8-float blocks."""
    for i in range(count):
        d = dst_base + i * (block_len + dst_stride) * 8
        s = src_base + i * (block_len + src_stride) * 8
        width = block_len * 8
        assert 0 <= s and s + width <= len(src)
        assert 0 <= d and d + width <= len(dst)
        if src is dst:
            # All uses in this candidate have globally disjoint source and
            # destination ranges, rather than depending on memmove behavior.
            assert d + width <= s or s + width <= d
        chunk = src[s:s + width]
        assert UNINITIALIZED not in chunk
        dst[d:d + width] = chunk


def merge_four(dst, src, rows, length):
    """MrgSort repeat mode: four contiguous equal-length sorted lists."""
    assert src is not dst
    row_slots = length * 4 * 2
    for r in range(rows):
        start = r * row_slots
        candidates = src[start:start + row_slots:2]
        assert UNINITIALIZED not in candidates
        for j in range(4):
            leaf = candidates[j * length:(j + 1) * length]
            assert leaf == sorted(leaf, key=order, reverse=True)
        merged = sorted(candidates, key=order, reverse=True)
        for i, score in enumerate(merged):
            dst[start + i * 2:start + i * 2 + 2] = [score, 0]


def generate_row(rng, experts, k, row, mode):
    if mode == "random":
        return [bits(rng.uniform(-100, 100)) for _ in range(experts)]
    if mode == "ties":
        return [bits(rng.choice((-math.inf, -10000., -1., 0., 1., 2., 10000.)))
                for _ in range(experts)]
    if mode == "all_negative_inf":
        return [NEG_INF] * experts
    if mode == "one_finite":
        result = [NEG_INF] * experts
        result[(row * 37 + experts - 1) % experts] = bits(1.)
        return result
    if mode == "cross_leaf_ties":
        return [bits(50. if i % 32 == 0 else -3.) for i in range(experts)]
    if mode == "fp32_edges":
        # Exact original FP32 adjacent values, subnormals, finite extrema,
        # and negative infinity. No conversion to half or approximate oracle.
        pool = (0xFF7FFFFF, 0xBF800001, 0xBF800000, 0x80000001,
                0x00000001, 0x3F800000, 0x3F800001, 0x7F7FFFFF, NEG_INF)
        return [pool[(i * 17 + row * 7) % len(pool)] for i in range(experts)]
    result = [bits(-100.)] * experts
    if mode == "tail_winners":
        # Force all winners into the final leaf, including partial E129 rows.
        # Where that leaf has fewer than k real inputs, put remaining winners
        # immediately before its boundary, exercising both final merge lists.
        for i in range(k):
            result[experts - 1 - i] = bits(40. + i)
    elif mode == "first_winners":
        # Concentrate the winners in one of the four merged leaves, rotating
        # between leaves so no implicit first-leaf preference can pass.
        start = (row % 4) * 32
        for i in range(k):
            result[start + i] = bits(60. + i)
    else:
        raise AssertionError(mode)
    return result


def check_batch(original, experts, k):
    rows = len(original)
    inner = ((experts + 31) // 32) * 32
    assert inner == 160
    a = [UNINITIALIZED] * (rows * inner * 2)
    b = [UNINITIALIZED] * (rows * inner * 2)
    for r, scores in enumerate(original):
        padded = scores + [NEG_INF] * (inner - experts)
        for leaf in range(5):
            sorted_leaf = sorted(padded[leaf * 32:(leaf + 1) * 32], key=order, reverse=True)
            for i, score in enumerate(sorted_leaf):
                offset = r * 320 + leaf * 64 + i * 2
                a[offset:offset + 2] = [score, 0]

    # These calls reproduce BuildFiveLeafShortPairs without algebraically
    # simplifying its offsets, so layout/padding mistakes remain observable.
    copy(b, 0, a, 0, rows * 5, 2, 6, 0)
    copy(a, 0, b, 0, rows, 8, 2, 0)
    copy(a, rows * 64, b, 64, rows, 2, 8, 0)
    merge_four(b, a, rows, 8)
    copy(a, 0, b, 0, rows, 2, 6, 2)
    # Whole ranges of this same-buffer operation are disjoint, including
    # batch1: destinations end at 32*rows, source starts at 64*rows.
    assert 32 * rows <= 64 * rows
    copy(a, 16, a, rows * 64, rows, 2, 0, 2)
    merge_four(b, a, rows, 4)

    for r, scores in enumerate(original):
        actual = b[r * 32:r * 32 + k * 2:2]
        expected = sorted(scores, key=value, reverse=True)[:k]
        assert actual == expected, {
            "rows": rows, "experts": experts, "k": k, "row": r,
            "actual_bits": actual, "expected_bits": expected,
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="prior inline proof's sampled sizes/E; full coverage is default")
    args = parser.parse_args()
    rng = random.Random(SEED)
    batches = (1, 2, 3, 7, 16, 31, 32, 47, 48) if args.quick else range(1, 49)
    columns = (129, 130, 136, 150, 151, 159, 160) if args.quick else range(129, 161)
    batch_count = row_count = score_count = 0
    for rows in batches:
        for experts in columns:
            for k in (6, 7):
                for mode in MODES:
                    original = [generate_row(rng, experts, k, r, mode) for r in range(rows)]
                    row_count += check_batch(original, experts, k)
                    batch_count += 1
                    score_count += rows * k
    print(json.dumps({
        "status": "PASS", "seed": SEED, "quick": args.quick,
        "batches_checked": batch_count, "rows_checked": row_count,
        "selected_score_bits_checked": score_count,
        "experts": list(columns), "batch_rows": list(batches), "k": [6, 7],
        "padded_columns": 160, "modes": MODES,
        "oracle": "original FP32 score bits, exact sorted top-k; no tolerance",
        "scope": "host copy-layout and exact selection; not CANN/device verification",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
