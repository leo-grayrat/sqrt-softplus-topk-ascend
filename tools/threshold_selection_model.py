"""Host-only exact selection contract and candidate-size experiment, not NPU timing."""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def select(row, sample_size, capacity=256, k=64):
    # NaN ordering is deliberately outside this model's contract.
    if any(math.isnan(x) for x in row):
        return None, None, "nan-fallback"
    sample = row[::len(row) // sample_size]
    threshold = sorted(sample, reverse=True)[k - 1]
    # >= preserves ties. Never truncate an overflowing candidate array.
    candidates = [x for x in row if x >= threshold]
    if len(candidates) > capacity:
        return sorted(row, reverse=True)[:k], len(candidates), "overflow-fallback"
    assert len(candidates) >= k
    return sorted(candidates, reverse=True)[:k], len(candidates), "selected"


def rows(seed):
    rng = random.Random(seed)
    for _ in range(1000):
        yield "normal", [rng.gauss(0, 1) for _ in range(1024)]
        yield "uniform", [rng.uniform(-20, 20) for _ in range(1024)]
        yield "ties", [float(rng.randrange(8)) for _ in range(1024)]
    yield "all-equal", [1.0] * 1024
    yield "ascending", list(map(float, range(1024)))
    yield "descending", list(map(float, reversed(range(1024))))
    yield "infinities", [float("inf")] * 64 + [float("-inf")] * 960
    yield "negative-infinity", [float("-inf")] * 1024
    # All winners in any one segment, and sample positions systematically biased.
    for start in range(0, 1024, 64):
        row = [-20.0] * 1024
        row[start:start + 64] = list(map(float, range(64)))
        yield "concentrated", row
    for stride in (2, 4):
        yield "sample-low", [-100.0 if i % stride == 0 else float(i) for i in range(1024)]
        yield "sample-high", [10000.0 + i if i % stride == 0 else float(i) for i in range(1024)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"seed": 20261004, "kind": "host correctness and candidate counts only", "configs": {}}
    for sample_size in (256, 512):
        stats = defaultdict(lambda: {"rows": 0, "fallbacks": 0, "candidate_sum": 0, "candidate_max": 0})
        for kind, row in rows(result["seed"]):
            selected, count, path = select(row, sample_size)
            assert selected == sorted(row, reverse=True)[:64], (sample_size, kind)
            stat = stats[kind]
            stat["rows"] += 1
            stat["fallbacks"] += path != "selected"
            stat["candidate_sum"] += count
            stat["candidate_max"] = max(stat["candidate_max"], count)
        assert select([float("nan")] + [0.0] * 1023, sample_size)[2] == "nan-fallback"
        result["configs"][str(sample_size)] = dict(stats)
        print(sample_size, "PASS", sum(s["rows"] for s in stats.values()), "rows;", dict(stats))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
