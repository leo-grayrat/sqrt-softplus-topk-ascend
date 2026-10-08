"""Check the top-4 merge identity independently of Ascend tensor operations."""
from itertools import combinations_with_replacement, product
import random


def merge4(a, b):
    x = [max(a[i], b[3 - i]) for i in range(4)]
    y = [max(x[0], x[2]), max(x[1], x[3]),
         min(x[0], x[2]), min(x[1], x[3])]
    return [max(y[0], y[1]), min(y[0], y[1]),
            max(y[2], y[3]), min(y[2], y[3])]


def network(values):
    groups = [sorted(values[i:i + 4], reverse=True)
              for i in range(0, len(values), 4)]
    while len(groups) > 1:
        groups = [merge4(groups[i], groups[i + 1])
                  for i in range(0, len(groups), 2)]
    return groups[0]


def main():
    lists = [list(reversed(x)) for x in
             combinations_with_replacement([-float('inf'), -1, 0, 1], 4)]
    for a, b in product(lists, repeat=2):
        assert merge4(a, b) == sorted(a + b, reverse=True)[:4]
    rng = random.Random(51)
    for _ in range(1000):
        values = [rng.randrange(-65504, 65505) for _ in range(256)]
        assert network(values) == sorted(values, reverse=True)[:4]
    print(f'PASS: {len(lists) ** 2} exhaustive merges; 1000 full rows')


if __name__ == '__main__':
    main()
