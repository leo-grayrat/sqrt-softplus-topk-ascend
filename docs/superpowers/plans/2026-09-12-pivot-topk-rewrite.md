# Pivot Top-K Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 exact pivot-filter 替换 FP32 `2<=k<=8` 的整行 Sort32/MrgSort 选择路径，使中大 E 有结构性提速可能，同时保留现有正确 fallback。

**Architecture:** 每行 logits 只搬入 UB 一次。先用 Compare/GetCmpMask 统计阈值两侧元素数量，把候选缩到 `k..64`；再用 GatherMask 收集候选，并仅对最多 64 个值做最终 Sort32/小 MrgSort。若固定迭代内无法得到安全候选集，则回退现有 prefix/NORMAL 路径。

**Tech Stack:** Ascend C / CANN 9.0.0 / dav-2201 / CPU twin-debug / CMake / C++14

**Spec:** `docs/superpowers/specs/2026-09-12-pivot-topk-rewrite-design.md`

## Global Constraints

- 所有实现只进入 `pivot-topk-rewrite` 分支，main 不动。
- 第一版 production 只替换 FP32 `2<=k<=8`；`k==1`、FP16、`k>8` 保留现有路径。
- 候选过滤必须是 exact：只有数学上保证包含全部 top-k 的集合才能进入最终排序。
- 候选上限为 64；固定 pivot 迭代上限后仍无法收敛则 fallback。
- 不做性能微实验树；完整 correctness 通过后只做一次 Judge 评价。
- 新提交文件只使用比赛允许的 `.asc` / `.h`。

---

### Task 1: Compare mask / popcount / GatherMask API probe

**Files:**
- Create: `debug/cpu_topk_probe/topk_pivot_mask_probe.asc`
- Modify: `debug/cpu_topk_probe/CMakeLists.txt`

**Interfaces:**
- Consumes: CANN 9.0 `Compare`, `GetCmpMask`, `GetBitCount<1>`, custom-pattern `GatherMask`.
- Produces: 一个 throwaway CPU twin-debug executable，证明 64 个 FP32 元素的一次 compare mask 的位序、popcount 和 GatherMask 自定义 pattern 对齐。

- [ ] **Step 1: 写 probe 的 host golden**

构造 64 个确定性 float，pivot 固定为 `0.5f`。Host 端计算 `value > pivot` 的布尔数组、计数和按原顺序压缩后的候选，作为期望结果。

- [ ] **Step 2: 写 device probe**

Device 端对同一 64 元素块执行 `Compare(..., CMPMODE::GT, mask=64, ...)`，`GetCmpMask` 保存 128B compare 结果；读取其中与 64 个 FP32 对应的 64 bits，用 `GetBitCount<1>` 得到计数；把位图按 GatherMask FP32 自定义 pattern 所需的 `uint32_t` 形式写入 pattern buffer，执行一次 `GatherMask`。

- [ ] **Step 3: 本地 clean build 验证**

Run:
```bash
cd debug/cpu_topk_probe
rm -rf build
cmake -S . -B build -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build build -j"$(nproc)"
./build/topk_pivot_mask_probe
```
Expected: 输出 compare popcount 与 host count 一致，GatherMask retained count 一致，候选逐元素一致，最后打印 `PASS: pivot mask probe`。

- [ ] **Step 4: 若接口/位布局不符，只修正 probe 到文档真实语义**

此阶段禁止改 production。若 probe 不能在 CANN 9.0 CPU twin-debug 闭环，则停止本路线并记录原因，不再猜 API。

- [ ] **Step 5: Commit**

```bash
git add debug/cpu_topk_probe/topk_pivot_mask_probe.asc debug/cpu_topk_probe/CMakeLists.txt
git commit -m "debug: verify pivot mask and gather semantics"
```

### Task 2: 为 pivot selector 先写 correctness oracle / regression cases

**Files:**
- Create: `tests/test_pivot_selector_reference.py`
- Modify: `reference/reference.py`

**Interfaces:**
- Consumes: `topk_first` / official reference semantics.
- Produces: `pivot_filter_reference(router_logits, k, candidate_limit=64, max_iters=8)`，用于验证算法数学条件，而不是模拟硬件性能。

- [ ] **Step 1: 先写失败测试**

测试集合固定覆盖：
```python
cases = [
    (65, 2), (95, 4), (127, 8), (128, 4),
    (257, 8), (511, 4), (1023, 8), (1024, 8),
]
```
并分别覆盖随机值、全相等值、重复阈值值、`+inf/-inf`。每例断言 `pivot_filter_reference(...)` 与 `topk_first(...)` 值完全对应（FP32 容差沿用现有 reference tests）。

- [ ] **Step 2: 运行并确认 RED**

Run:
```bash
pytest -q tests/test_pivot_selector_reference.py
```
Expected: 因 `pivot_filter_reference` 尚不存在而失败。

- [ ] **Step 3: 实现最小 Python exact pivot-filter reference**

算法只做：维护 `lo/hi`，pivot=`(lo+hi)/2`；计算 `count_gt/count_ge`；仅在 `k <= candidates <=64` 时排序候选；固定 8 次后 fallback 到 `topk_first`。全相等或无有限区间时允许立即 fallback。

- [ ] **Step 4: 运行并确认 GREEN**

Run:
```bash
pytest -q tests/test_pivot_selector_reference.py tests/test_reference.py
```
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add reference/reference.py tests/test_pivot_selector_reference.py
git commit -m "test: define exact pivot topk reference"
```

### Task 3: 实现 Ascend FP32 pivot selector

**Files:**
- Create: `submission/pivot_topk.h`
- Create: `debug/cpu_topk_probe/topk_pivot_selector_probe.asc`
- Modify: `debug/cpu_topk_probe/CMakeLists.txt`

**Interfaces:**
- Produces: `SqrtSoftplusPivotTopKKernel`，输入 `(rows, experts, k)`，只输出排序后的 top-k scores；内部失败条件调用 caller 提供的旧路径，而不是输出近似结果。

- [ ] **Step 1: 先写 selector probe 的失败 host checker**

`topk_pivot_selector_probe.asc` 先引用尚不存在的 `submission/pivot_topk.h`，并准备 rows=`{1,3}`、E=`{65,95,128,257,511,1023,1024}`、k=`{2,4,8}` 的确定性随机/重复/Inf 输入，host 端 full-sort 得到 golden。

- [ ] **Step 2: clean build 确认 RED**

Run:
```bash
cd debug/cpu_topk_probe
rm -rf build
cmake -S . -B build -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build build -j"$(nproc)"
```
Expected: 因 `pivot_topk.h` / selector 未实现而编译失败。

- [ ] **Step 3: 实现单行 pivot scan**

`pivot_topk.h` 中每行一次搬入 UB；先 `ReduceMin/ReduceMax` 得到初始值域。每轮按 64 个 FP32 一组比较 `>pivot` 和 `>=pivot`，mask 尾部严格屏蔽 padding；使用 `GetBitCount<1>` 累加 `count_gt/count_ge`。最多 8 轮。

- [ ] **Step 4: 实现安全候选收集**

仅当以下任一条件成立时收集：
- `count_gt >= k && count_gt <= 64`：收集 `>pivot`；
- `count_gt < k && k <= count_ge && count_ge <=64`：收集 `>=pivot`。

按每个 64-element block 的 compare mask 构造 GatherMask pattern，连续 append 到最多 64 个 float 的候选 buffer。其余状态继续缩值域或标记 fallback。

- [ ] **Step 5: 实现最多 64 候选的 exact 排序**

候选 pad 到 32 或 64：
- `<=32`：一次 `Sort32`；
- `33..64`：两次 `Sort32` 后一次 2-way `MrgSort`。
只提取 value，不生成整行 E 的 index。前 k value 复用 `TransformSelected` 数学变换。

- [ ] **Step 6: selector probe GREEN**

Run:
```bash
./build/topk_pivot_selector_probe
```
Expected: 所有 rows/E/k 与重复值/Inf case PASS；输出类似 `PASS: <N> pivot selector cases`。

- [ ] **Step 7: Commit**

```bash
git add submission/pivot_topk.h debug/cpu_topk_probe/topk_pivot_selector_probe.asc debug/cpu_topk_probe/CMakeLists.txt
git commit -m "feat: add exact pivot topk selector"
```

### Task 4: 接入 production，并一次性做完整 correctness / Judge

**Files:**
- Modify: `submission/kernel.asc`
- Modify: `debug/cpu_topk_probe/topk_prefix_production_probe.asc`（重命名说明可不做，沿用现有 production checker）
- Delete from production use only: `submission/dense_repeat_override.h` 可保留历史文件，但 `kernel.asc` 不再 include/dispatch 它。

**Interfaces:**
- FP32 `2<=k<=8` 且 E 达到第一版 gate 时走 pivot selector；无法安全收敛必须走原 prefix fallback。
- `k==1`、FP16、`k>8` 行为不变。

- [ ] **Step 1: production checker 先加 pivot 命中形状并确认旧代码仍只是旧路径**

至少覆盖 E=`{65,95,127,128,257,384,511,512,640,768,896,1023,1024}`，k=`{2,4,8}`，rows=`{1,3}`，以及重复值/Inf 特例。

- [ ] **Step 2: 接入 production dispatch**

`kernel.asc` include `pivot_topk.h`；撤掉 dense-repeat macro override。FP32 `2<=k<=8` 调 pivot dispatcher，dispatcher 内部保持旧 `SqrtSoftplusPrefixMergeTopKKernel` 作为 fallback。

- [ ] **Step 3: clean build + 完整 CPU correctness**

Run:
```bash
cd debug/cpu_topk_probe
rm -rf build
cmake -S . -B build -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build build -j"$(nproc)"
./build/topk_pivot_mask_probe
./build/topk_pivot_selector_probe
./build/topk_prefix_production_probe
```
Expected: 三个 executable 全 PASS，无 checker warning / abort。

- [ ] **Step 4: 一次 CANNJudge**

记录 15 个 case。判断只看相对当前约 `2.37,3.89,252.11,8.09,17.07,24.12,59.09,87.29,96.82,244.48,252.67,650.03,634.57,2130,3460 us` 的变化。

继续标准：中大点出现约 2× 或更强的稳定下降趋势。若仍只有几十个百分点以内，则关闭该路线并恢复 main，不追加微调实验。

- [ ] **Step 5: Commit results**

将 correctness 与 Judge 数据写入本分支实验记录，并提交；只有达到继续标准才考虑合并回 main。
