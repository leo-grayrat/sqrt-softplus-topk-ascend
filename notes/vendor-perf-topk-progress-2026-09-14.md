# Vendor Perf TopK 进展记录（2026-09-14）

## 背景

当前任务为 `SqrtSoftplusTopK`：输入 `router_logits(T,E)`，`E=64..1024`，`k=1..64`，FP16/FP32，目标 dav-2201。由于 `sqrt(softplus(x))` 单调递增，可以先在 raw logits 上做 exact Top-K，再只变换选中值；最终只输出 scores。

此前 bulk-prefix packing、跨行 batching、repeated ReduceMax 等方向均未在真实 Judge 上形成稳定收益。随后转向直接参考 Huawei `hicann/ops-transformer` 中 A2/A3 的 `MoeGatingTopKSoftmaxPerf` 执行结构。

参考源码：

- `moe/moe_gating_top_k_softmax/op_kernel/moe_gating_top_k_softmax_perf.h`
- `moe/moe_gating_top_k_softmax/op_host/moe_gating_top_k_softmax_tiling_perf.cpp`

## 第一轮：移植 vendor Perf selection 结构

第一版 production 仍只覆盖 FP32、`k<=8`、`E%32==0`，核心结构：

- 多行一次 `Sort32`；
- 每 32 个元素固定保留 top-8 pair；
- 跨行批量 prefix copy；
- 分层 `MrgSort`；
- 输入 queue 双缓冲；
- 最终只抽 scores，不保留 index 输出。

真实 Judge 中最明显变化为 case 7：

`59.18 us -> 30.69 us`，约 1.93x，加速约 48%。

其余 case 基本保持原水平。这是目前第一个在真实 Judge 上出现结构级收益的方向，证明 vendor Perf 的 selection 执行组织值得继续跟进。

## 第二轮：自适应 batchRows

将固定 `vendorBatchRows=16/8/4` 和“必须攒满 batch 才进入新路径”的 gate 改为：

```text
capacity = 16 / 8 / 4
rowsPerCore = ceil(rows / launchedBlockNum)
batchRows = min(capacity, rowsPerCore)
groupCount = ceil(rows / batchRows)
activeBlockNum = min(groupCount, launchedBlockNum)
```

真实 Judge 基本没有进一步收益：case 7 `30.69 -> 30.46 us`，其余点仅噪声级变化。

结论：3/9/13 没有受益并不是因为固定 batch gate 卡住。

## 第三轮：去掉 E%32 限制，UB 内 padding 到 inner

GM 仍按真实 `E` 紧密存放；进入 UB 后按：

`inner = ceil(E/32) * 32`

补 `-inf`，后续 Sort32 / top-8 prefix / merge 全部按 `inner` 处理。这样非 32 对齐 E 也能进入同一 vendor Perf 路径。

真实 Judge：

```text
case 3 : 253.15 -> 151.81 us   -40.0%
case 7 :  30.46 ->  30.58 us   持平
case 9 :  96.13 ->  48.66 us   -49.4%
case 13: 630.75 -> 285.57 us   -54.7%
```

这说明 3/7/9/13 基本可以视为同一 FP32 small-k 优化族。此前 3/9/13 没动，主要是因为它们没有进入 `E%32==0` 的 vendor Perf 路径。

与前几名正常队伍的典型水平相比，目前仍约有：

```text
case 3 : 151.81 / ~42.8  ~= 3.5x
case 7 :  30.58 / ~9.6   ~= 3.2x
case 9 :  48.66 / ~13.8  ~= 3.5x
case 13: 285.57 / ~62.2  ~= 4.6x
```

TBest 只作为旁证，不再作为统一性能目标；尤其 case 3 的 4.36 us 明显偏离前十名约 42--44 us 的正常分布。

## 与 Huawei Perf 源码仍存在的主要差异

### 1. CopyIn / layout staging

Huawei Perf 的 `CopyIn()` 对一个 UB batch 只发一次 `DataCopyPad`：

- 列宽已经满足搬运对齐时，可以把 `curRowsNum * col` 当成连续块；
- 否则用 `blockCount=curRowsNum` 一次搬多行。

其输入计算布局使用 32B 对齐行宽，随后通过单独的 `Rearrange()` 把每行重排成 TopK 所需的 32 元素对齐 `colAlign`。源码明确注释：softmax 阶段按 32B 对齐，而 TopK 要求 32-element aligned。

当前实现跳过这层 staging/rearrange，因此 `CopyInGroup()` 仍在 `for(slot...)` 中逐行 `DataCopyPad` 到 `slot * inner_`。

### 2. 尾 batch 的 curRowsNum

Huawei Perf 把真实 `curRowsNum` 一直传给 Sort32 / prefix copy / MrgSort / GatherMask；尾 batch 只计算有效行。

当前实现虽然最后只写 `validRows`，但 Sort32、merge、Gather 仍按完整 `batchRows_` 处理 dummy rows。

### 3. merge packing 已经较接近 vendor

Huawei `MergeSortFP32PerfCopy` 同样用 `blockCount=curRowsNum` + stride 跨行抽每个 Sort32 chunk 的 top-8。当前实现这一部分已经较接近，不优先重写 merge 树。

## 已批准的下一步

保持 selection 数学结构不变，只继续贴近 vendor Perf 的输入与尾 batch 执行组织：

1. 增加按 32B 对齐行宽的 staging input buffer；
2. 一个 batch 只发一次 `DataCopyPad` 搬入 staging；
3. 批量 rearrange 到当前 `inner=ceil(E/32)*32` 的 TopK 布局，并补 `-inf`；
4. `ComputeGroup()` 将真实 `validRows` 传给 Sort32 / prefix copy / MrgSort / GatherMask，尾 batch 不再处理 dummy rows；
5. 自适应 batch capacity、FP32 `k<=8` dispatch、top-8 prefix + merge 树全部保持不变；
6. 完成 CPU correctness 后直接跑完整 Judge，不增加 simulator 性能准入。

目标不是再证明“vendor Perf 是否值得做”——这一点已经由 case 3/7/9/13 的真实 Judge 结果证明。下一轮只回答：补齐 vendor 的 batch 搬运 / rearrange / curRowsNum 后，是否还能继续缩小目前约 3--5x 的正常前排差距。
