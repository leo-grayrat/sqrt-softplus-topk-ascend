# Rows scaling status

Prepared on `profile-topk-cost-model`:

- `tools/profile_harness/main.cpp`: optional `blockNum`, default unchanged (`rows`).
- `tools/check_rows_scaling_harness.py`: static guard for blockNum propagation.
- `tools/profile_rows_scaling.sh`: rows 1/2/4/8, fixed `blockNum=1`, `E=1024,k=8` by default.
- `tools/summarize_rows_scaling.py`: pipe totals, scalar hotspots, linear fixed/per-row fits.
- `tools/check_rows_scaling_scripts.py`: guards the experiment invariants.

Pending: run in the user's CANN 9.0 simulator environment. Do not infer runtime results before that run.
