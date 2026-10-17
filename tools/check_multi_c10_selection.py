#!/usr/bin/env python3
"""Exact C10 raw-score and byte-offset proof; no device/transform claims."""
import json
import math
import random
import struct

SEED = 831016


def half_float(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]


def float_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def as_float(x):
    return struct.unpack('<f', struct.pack('<I', x))[0]


def key(x):
    return (~x & 0xffffffff) if x & 0x80000000 else x ^ 0x80000000


def make_row(rng, row, mode):
    scores = [half_float(rng.uniform(-20, 20)) for _ in range(128)]
    if mode < 4:
        scores = [-20.] * 128
        if mode == 0:
            for i in range(16): scores[((row // 8) % 2) * 64 + i] = 40. + i
        elif mode == 1:
            for i in range(8):
                g = (row * 13 + i * 7) % 64
                scores[g], scores[g + 64] = 40. + i, 55. + i
        elif mode == 2:
            for i in range(16): scores[(row * 11 + i * 3) % 64 + (i % 2) * 64] = 40. + i
        else:
            for i in range(16): scores[127 - i] = 40. + i
    elif mode == 4: scores = [-4.] * 128
    elif mode == 5: scores = [-math.inf] * 128
    elif mode == 6: scores = [half_float(1. + (i % 3) * 2**-10) for i in range(128)]
    return [float_bits(x) for x in scores]


def copy(dst, src, count, length, src_stride, dst_stride):
    for i in range(count):
        s, d = i * (length + src_stride) * 8, i * (length + dst_stride) * 8
        chunk = src[s:s + length * 8]
        assert None not in chunk
        dst[d:d + length * 8] = chunk


def check(original):
    rows = len(original)
    table = ((rows * 32 + 63) // 64) * 64
    index = list(range(rows * 64))
    # Match unsigned shifts exactly, including vector-table padding.
    gather = [((j >> 5) << 8) + ((((j << 27) & 0xffffffff) >> 28) << 3) + 4
              for j in index[:table]]
    base = [((j >> 5) << 7) + (((j << 31) & 0xffffffff) >> 25)
            for j in index[:table]]
    for j in range(rows * 32, table):
        gather[j], base[j] = 4, 0
    assert all(0 <= x < rows * 64 * 4 for x in gather)
    index = [((j << 26) & 0xffffffff) >> 26 for j in index]
    a = [None] * (rows * 128)
    b = [None] * (rows * 128)
    for r, scores in enumerate(original):
        maxima = [max(scores[g], scores[g + 64], key=key) for g in range(64)]
        for leaf in range(2):
            pairs = [(maxima[g], index[r * 64 + g]) for g in range(leaf * 32, leaf * 32 + 32)]
            pairs.sort(key=lambda pair: key(pair[0]), reverse=True)
            for j, pair in enumerate(pairs):
                s = r * 128 + leaf * 64 + j * 2
                a[s:s + 2] = pair
    copy(b, a, rows * 2, 4, 4, 0)
    for r in range(rows):
        pairs = [tuple(b[r * 64 + j * 2:r * 64 + j * 2 + 2]) for j in range(32)]
        for leaf in range(4):
            part = pairs[leaf * 8:(leaf + 1) * 8]
            assert part == sorted(part, key=lambda p: key(p[0]), reverse=True)
        pairs.sort(key=lambda p: key(p[0]), reverse=True)
        assert len(set(p[1] for p in pairs[:16])) == 16
        for j, pair in enumerate(pairs): a[r * 64 + j * 2:r * 64 + j * 2 + 2] = pair
    flattened = [score for scores in original for score in scores]
    packed = []
    for j in range(rows * 32):
        assert gather[j] == (j // 32) * 256 + ((j % 32) // 2) * 8 + 4
        assert base[j] == (j // 32) * 128 + (j % 2) * 64
        group_id = a[gather[j] // 4]
        offset_bytes = (group_id + base[j]) << 2
        assert 0 <= offset_bytes < rows * 128 * 4
        packed.append(flattened[offset_bytes // 4])
    for r, scores in enumerate(original):
        actual = sorted(packed[r * 32:(r + 1) * 32], key=key, reverse=True)[:16]
        expected = sorted(scores, key=as_float, reverse=True)[:16]
        assert actual == expected, (rows, r, actual, expected)


def main():
    rng = random.Random(SEED)
    # The production round-robin loop visits each batch exactly once. Check
    # balanced capacities, tails, and per-core coverage independently of values.
    partition_cases = 0
    examples = []
    for total in list(range(1, 1025)) + [4096, 16384, 16403]:
        for cores in [1, 2, 3, 8, 24, 32]:
            per_core = (total + cores - 1) // cores
            batches_per_core = (per_core + 47) // 48
            batch = (per_core + batches_per_core - 1) // batches_per_core
            assert 1 <= batch <= 48
            groups = (total + batch - 1) // batch
            active = min(groups, cores)
            loads = [sum(min(batch, total - g * batch)
                         for g in range(core, groups, active))
                     for core in range(active)]
            assert all(load > 0 for load in loads)
            assert sum(loads) == total
            partition_cases += 1
            if total in [4096, 16384] and cores == 32:
                examples.append({'rows': total, 'cores': cores, 'batch': batch,
                                 'busiest_core_rows': max(loads)})
    batches = checked_rows = 0
    for rows in range(1, 49):
        for mode in range(8):
            original = [make_row(rng, r, mode) for r in range(rows)]
            check(original)
            batches += 1
            checked_rows += rows
    print(json.dumps({'status': 'PASS', 'seed': SEED, 'batches_checked': batches,
                      'rows_checked': checked_rows, 'selected_score_bits_checked': checked_rows * 16,
                      'batch_rows_range': [1, 48], 'experts': 128, 'k': 16,
                      'modes': 8, 'oracle': 'original half inputs represented as exact FP32 bits',
                      'partition_cases_checked': partition_cases,
                      'balanced_partition_examples': examples,
                      'scope': 'exact raw selection and unsigned byte-index formulas; no device claim'}, sort_keys=True))


if __name__ == '__main__':
    main()
