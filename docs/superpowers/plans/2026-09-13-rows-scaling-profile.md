# Single-Block Rows Scaling Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the dominant simulator SCALAR cost is one-time kernel/block setup or repeated per processed row by forcing one block to process 1/2/4/8 rows at `E=1024,k=8`.

**Architecture:** Extend the existing native ASC profile host driver with an optional `blockNum` argument while preserving the current default `blockNum=rows`. Add a profiling script that always launches with `blockNum=1` and profiles rows 1/2/4/8 using the production dense baseline. Add a summarizer that aggregates instruction CSVs, reports pipe totals and scalar hotspots, and fits `cycles = fixed + per_row * rows`.

**Tech Stack:** C++ host harness, Ascend C native ASC CMake, `msprof op simulator`, Bash, Python CSV analysis.

**Spec:** This plan is the executable spec; the experiment design is the rows-scaling discussion recorded after bulk-prefix-pack showed simulator-only barrier savings with no Judge gain.

## Global Constraints

- Profiling only; do not change production submission code.
- Fixed shape: `E=1024`, `k=8`.
- Rows: `1,2,4,8`.
- Force `blockNum=1` for every run so the same vector core processes all rows sequentially.
- Reuse the proven native ASC/msprof harness and simulator library path handling.
- Do not interpret simulator absolute wall time as Judge time.

---

### Task 1: Make block count explicit in the profile harness

**Files:**
- Create: `tools/check_rows_scaling_harness.py`
- Modify: `tools/profile_harness/main.cpp`

**Interfaces:**
- Consumes: existing `LaunchProfileKernel(..., blockNum)`.
- Produces: CLI `topk_profile_one rows E k [blockNum]`; omitted `blockNum` retains the old `blockNum=rows` behavior.

- [ ] Add a static regression check that requires optional blockNum parsing, validation, and propagation.
- [ ] Verify the check fails on the old harness.
- [ ] Implement optional `blockNum` in `main.cpp`.
- [ ] Verify the static check passes and existing 3-argument invocation remains valid.

### Task 2: Add the single-block rows sweep

**Files:**
- Create: `tools/profile_rows_scaling.sh`

**Interfaces:**
- Consumes: `tools/profile_harness`, production `origin/main` submission sources, `msprof op simulator`.
- Produces: `profile-output/<label>_rowscaling_e1024_k8/rows_{1,2,4,8}`.

- [ ] Generate the same production dense dispatcher source used by the established profiling harness.
- [ ] Build once in simulator mode.
- [ ] Run rows 1/2/4/8 with CLI blockNum fixed to 1.
- [ ] Require an instruction CSV for every rows value.

### Task 3: Summarize fixed versus per-row cost

**Files:**
- Create: `tools/summarize_rows_scaling.py`

**Interfaces:**
- Consumes: four rows profile directories.
- Produces: per-row pipe totals, selected scalar instruction totals, and least-squares `fixed + per_row * rows` fits for SCALAR/VECTOR and scalar hotspots.

- [ ] Aggregate all `*_instr_exe.csv` files under the latest `OPPROF_*` for each rows value.
- [ ] Report SCALAR, VECTOR, MTE2, MTE3 totals.
- [ ] Report `ST_XD_XN_IMM`, `STI_XN_IMM`, `STP_XI_XJ_XN`, `LD`, and other top scalar instructions.
- [ ] Fit and print fixed/per-row cycles and `R²` for SCALAR and VECTOR totals.
- [ ] Print a simple interpretation guard: high R² with large per-row term means repeated per-row control; large fixed intercept with small slope means setup-dominated.

### Task 4: Verification

**Files:**
- Verify: `tools/check_rows_scaling_harness.py`
- Verify: `tools/profile_rows_scaling.sh`
- Verify: `tools/summarize_rows_scaling.py`

- [ ] Run the static harness check.
- [ ] Run `bash -n tools/profile_rows_scaling.sh`.
- [ ] Run `python3 -m py_compile` on both Python tools.
- [ ] Do not claim CANN/msprof runtime success until the user's CANN environment runs the script and returns all four profiles.
