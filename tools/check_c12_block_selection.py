"""Check the exact block-max selection theorem, including ties."""
import itertools
import random


def select(values, block, k):
    groups = [values[j:j + block] for j in range(0, len(values), block)]
    chosen = sorted(range(len(groups)), key=lambda j: max(groups[j]), reverse=True)[:k]
    return sorted((x for j in chosen for x in groups[j]), reverse=True)[:k]


count = 0
for values in itertools.product([float('-inf'), -1.0, 0.0, 1.0], repeat=6):
    for k in (1, 2, 3):
        assert select(values, 2, k) == sorted(values, reverse=True)[:k]
        count += 1

rng = random.Random(20261012)
for case in range(5000):
    values = [rng.choice([-65504.0, -4.0, 0.0, 1.0, 65504.0, float('-inf')])
              if case % 2 else rng.uniform(-100, 100) for _ in range(256)]
    if case % 5 == 0:
        start = rng.randrange(32) * 8
        values[start:start + 4] = [1000.0] * 4
    for k in (2, 3, 4):
        assert select(values, 8, k) == sorted(values, reverse=True)[:k]
        count += 1
print(f'PASS: {count} exact block-selection comparisons, including ties')
