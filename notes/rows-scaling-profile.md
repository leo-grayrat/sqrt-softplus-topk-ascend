# Single-block rows scaling profile

## Question

After bulk-prefix packing removed most simulator VECTOR barrier cycles but produced no visible Judge gain, determine whether the remaining dominant SCALAR cost is mostly:

1. fixed kernel/block setup cost, or
2. repeated per-row control / tensor bookkeeping cost.

## Controlled experiment

Use the established production dense baseline from `origin/main` and the native ASC/msprof harness.

Fixed:

- `E=1024`
- `k=8`
- `blockNum=1`

Sweep:

- `rows=1`
- `rows=2`
- `rows=4`
- `rows=8`

Forcing `blockNum=1` is essential. The same vector core must execute `ProcessRow()` repeatedly; otherwise increasing rows would also increase the number of blocks and make fixed block setup indistinguishable from per-row work.

## Primary measurements

- SCALAR total calls / cycles / time
- VECTOR total calls / cycles / time
- MTE2 / MTE3 totals
- dominant SCALAR instructions, especially `ST_XD_XN_IMM`, `STI_XN_IMM`, `STP_XI_XJ_XN`, `LD`

Fit each cycle series as:

```text
cycles(rows) = fixed + per_row * rows
```

Interpretation:

- high linearity + large `per_row * 8` share: repeated per-row overhead dominates; next experiment should batch vector work across rows / reduce per-row API and tensor bookkeeping;
- high linearity + large fixed intercept: setup dominates; next experiment should amortize setup over more rows per block;
- mixed/non-linear: inspect instruction-level scaling before changing production code.

## Commands

```bash
cd ~/sqrt-softplus-topk-ascend
git switch profile-topk-cost-model
git pull --ff-only
source /home/leo/Ascend/cann-9.0.0/set_env.sh

python3 tools/check_rows_scaling_harness.py
bash tools/profile_rows_scaling.sh Ascend920A 1024 8 dav-2201
python3 tools/summarize_rows_scaling.py \
  profile-output/Ascend920A_rowscaling_e1024_k8
```

This experiment is profiling-only. Do not submit it to Judge.
