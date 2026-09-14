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

## 第四轮：补齐 vendor 的 staging 与 validRows

第四轮进一步贴近 Huawei Perf 的输入与尾 batch 执行组织：

- 增加按 32B 对齐行宽的 staging input buffer；
- 一个 batch 一次 `DataCopyPad`；
- 非 32-element 对齐时批量 rearrange 到 `inner=ceil(E/32)*32`；
- Sort32 / prefix copy / MrgSort / GatherMask 只处理真实 `validRows`。

真实 Judge 15/15 Pass：

```text
case 3 : 154.49 us
case 7 :  29.97 us
case 9 :  46.68 us
case 13: 273.36 us
```

相对第三轮仅有小幅收益或噪声，说明 staging / tail-batch 组织值得保留，但不是当前主要瓶颈。

## 第五轮：批量 selected-value 后处理

保持 selection / merge / staging / dispatch 不变，只重组 `GatherMask` 之后的后处理：

- 跨行把每行前 8 个值压紧成 `[validRows, 8]`；
- 原先每行一次 `TransformSelected(8)` 改成整个 batch 一次 `TransformSelected(validRows * 8)`；
- `k=8` 时可顺带整批 copyout；`k<8` 仍逐行 copyout。

真实 Judge 15/15 Pass：

```text
case 3 : 139.62 us
case 7 :  18.69 us
case 9 :  28.38 us
case 13: 172.63 us
```

相对第四轮：

```text
case 3 :  -9.6%
case 7 : -37.6%
case 9 : -39.2%
case 13: -36.8%
```

这是继第三轮后第二次明确的结构级真实收益。

### A/B：batch copyout 不是主因

`vendor-perf-topk-ab-row-copyout` 只撤销 `k=8` 整批 copyout，其余第五轮逻辑不变：

```text
case 3 : 140.21 us
case 7 :  18.70 us
case 9 :  28.04 us
case 13: 177.88 us
```

与第五轮基本重合。结论：GM 输出方式不是第五轮主要收益来源，不继续围绕 copyout 深挖。

### A/B：batched transform 是主要收益来源

`vendor-perf-topk-ab-row-transform` 保留 packed layout 与 copyout，只把一次 `TransformSelected(validRows * 8)` 恢复成逐行 `TransformSelected(8)`：

```text
case 3 : 156.20 us
case 7 :  27.67 us
case 9 :  43.05 us
case 13: 268.87 us
```

相对第五轮退化约 `+11.9% / +48.0% / +51.7% / +55.8%`，并接近第四轮。

因此可以确认：第五轮 30--40% 级的大头来自把大量极短的 `Abs -> Muls -> Exp -> Adds -> Ln -> Maxs -> Add -> Sqrt` 向量操作从“每行发射一次”合并为跨行较长向量操作。case 7/9 仍比第四轮约快 7--8%，说明 packed 后处理布局本身可能还有少量残余贡献。

## 当前判断与下一步

1. production 保留第五轮 batched `TransformSelected(validRows * 8)`。
2. batch copyout 降级，不再投入。
3. simulator 仍只用于机制诊断，Judge 才是性能结论来源。
4. 下一轮优先检查后处理是否还存在可跨行合并的短 vector op，以及当前始终变换 `top-8` 是否在 `k<8` shape 上做了不必要工作；若能在不引入高额 compaction 成本的前提下只变换真实 k 个值，应做独立 production spike。
5. 若后处理进一步压缩收益有限，再回到 vendor selection 路径寻找剩余 2--3x 差距的结构来源。
