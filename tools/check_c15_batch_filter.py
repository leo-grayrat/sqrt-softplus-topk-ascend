"""Independent full-row oracle for guarded flat compaction and row restoration."""
import json
import numpy as np

rng = np.random.default_rng(85017)
checked = fast = fallback = 0
for rows in range(1, 11):
    for trial in range(128):
        x = rng.standard_normal((rows, 1024), dtype=np.float32)
        threshold = np.float32(rng.uniform(-3, 3))
        if trial < 8:
            count = [0, 63, 64, 65, 127, 128, 129, 1024][trial]
            x.fill(-np.inf)
            for row in range(rows):
                positions = rng.choice(1024, count, replace=False)
                x[row, positions] = rng.uniform(1, 8, count).astype(np.float32)
            threshold = np.float32(0)
        elif trial < 32:
            x.fill(-np.inf)
            for row in range(rows):
                count = int(rng.integers(64, 129))
                positions = rng.choice(1024, count, replace=False)
                x[row, positions] = rng.choice(np.array([1, 2, 3, np.inf], np.float32), count)
            threshold = np.float32(1)
        mask = x >= threshold
        counts = mask.sum(axis=1)
        expected = np.sort(x, axis=1)[:, ::-1][:, :64].copy()
        if np.all((counts >= 64) & (counts <= 128)):
            flat = np.full(rows * 512, -np.inf, np.float32)
            compact = x[mask]
            flat[:len(compact)] = compact
            prefix = np.concatenate(([0], np.cumsum(counts)[:-1]))
            indices = prefix[:, None] + np.arange(128)[None, :]
            assert indices.max() < rows * 128 <= len(flat)
            restored = flat[indices]
            restored[np.arange(128)[None, :] >= counts[:, None]] = -np.inf
            actual = np.sort(restored, axis=1)[:, ::-1][:, :64].copy()
            assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
            fast += rows
        else:
            fallback += rows
        checked += rows
print(json.dumps({'status': 'PASS', 'rows_checked': checked,
                  'fast_rows_exact_bits': fast, 'guard_rejected_rows': fallback,
                  'batch_range': [1, 10], 'seed': 85017,
                  'scope': 'conditional exactness/address proof, not device or speed evidence'}))
