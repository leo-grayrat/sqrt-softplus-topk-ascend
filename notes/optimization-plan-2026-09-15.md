# 后续优化计划（2026-09-15）

## 目标变化

此前测试点 shape 隐藏，只能根据哪些 case 随某个 gate 变化来反推 workload。现在已有高可信的 15 点 `(T,E,k,dtype)` 信息，后续不再优先做全局盲调，而按**覆盖范围 × 当前耗时 × 可复用已有结构**排序。

当前 `main` 固定在 Round 6，后续每个方向都从 main 开独立短期实验。只有真实 Judge 证明值得保留的结果才回 main。

## 优先级 1：把 vendor-perf 扩到 FP16 small-k

直接目标：case 12；同时覆盖 case 2 / 4。

```text
case 2    T=16     E=97   k=5   fp16     3.77 us
case 4    T=64     E=256  k=4   fp16     7.80 us
case 12   T=32768  E=256  k=3   fp16   650.44 us
```

case 12 是目前最显眼的“已有成熟算法结构却没有覆盖”的点：`E=256,k=3` 完全属于 small-k，只因 dtype=fp16 仍走旧通用路径。

### 第一阶段不直接大改 production

先回答一个 API / layout 问题：Huawei `MoeGatingTopKSoftmaxPerf` 本身是 `template <typename T>`，但我们当前移植版把输入、Sort32 pair、GatherMask 和输出布局都固定成 float。需要确认 dav-2201 下 half 的：

- Sort32 pair 实际布局；
- MrgSort 输入输出 stride；
- GatherMask value pattern；
- 是否可以保持 half selection，最后一次性 Cast 到 FP32 做 batched transform。

先做最小 half vendor probe，固定接近 case 12 的 `E=256,k=3`。如果 half 原生路径接口成立，再把 `SqrtSoftplusVendorPerfTopKKernel` 模板化；如果 half pair 路径不合适，第二选择才是“GM half -> UB cast float -> 复用 FP32 vendor selection”。

### 成功标准

优先看 case 12 是否出现结构性下降。case 2/4 太小，只用于确认没有明显回退，不拿它们决定方向。

如果 case 12 仍只是个位数百分比改善，则停止继续精修 FP16 small-k；如果出现接近当前 FP32 vendor 路径曾经得到的倍数级收益，则立即合回 main。

## 优先级 2：固定 keep=16 的 vendor-style medium-k 路径

目标 workload：

```text
case 5    T=512    E=200  k=16  fp16    16.63 us
case 6    T=1024   E=130  k=15  fp32    23.95 us
case 10   T=16384  E=128  k=16  fp16   244.10 us
case 14   T=65536  E=512  k=10  fp16     2.13 ms
```

这里最值得的是 case 14，其次是 case 10。现有 vendor-perf 每个 32 元素块固定保留 top-8；对 `k=9..16` 可以自然改成固定保留 top-16，再沿用多行 Sort32、分层 MrgSort、batched extraction/transform 的总体结构。

这比立刻研究新的 selection 算法更合理，因为我们已经知道在 A2/A3 上真正有效的是执行组织：跨行批量发射、prefix merge、减少短调用和重复控制开销。

### 设计约束

- 第一版只做 `9<=k<=16`，不同时推广到 32/64；
- keep 固定 16，避免动态 prefix 宽度引入过多分支；
- 重新做 UB 账本和 batch capacity，不能照搬 keep=8 的 16/8/4；
- FP32 与 FP16 可以分两步验证，优先选择能覆盖 case 14/10 的方案；
- transform 继续跨行 batch，不退回逐行调用。

### 成功标准

case 10/14 至少一个出现明显结构性收益，否则不继续把 keep-width 往 32/64 推。

## 优先级 3：large-k 单独设计，不机械放大 keep 宽度

目标：

```text
case 8    T=4096   E=256   k=64  fp16    86.99 us
case 11   T=16384  E=97    k=32  fp32   252.30 us
case 15   T=65555  E=1024  k=64  fp32     3.46 ms
```

这三点中 case 15 的绝对耗时最大，但 `k=64` 已经不是 small-prefix 场景。不能预设“把 keep=8 改成 keep=64”一定成立。

先比较两类结构：

1. Sort32 后做固定宽度 prefix merge tree，只在子树合并后截断到 k；
2. 继续使用通用 TopK selection，但把当前回退路径中的 post-transform / copyout / 多行组织改成 batch 化，先消除已知固定开销。

case 11 很特殊：`E=97,k=32`，k/E 已接近三分之一；case 15 则是 `E=1024,k=64`。两者可能不应共用同一 selection 策略。

large-k 方向必须在真实 shape 上先做成本账本，再选实现，不先开“通吃 k<=64”的大工程。

## 优先级 4：最后再做 FP32 small-k shape-aware batch tuning

已知：

```text
case 3   inner=896, k=8: batch 4 -> 5 有约 5% 收益
case 7   inner=160, k=7: batch 16 -> 32 明显退化
case 9   inner=160, k=6: batch 16 -> 32 基本持平/略好
case 13  inner=224, k=8: batch 16 -> 22 小幅改善
```

这说明 `inner` 不是 batch 最优点的唯一变量。后续若仍值得压榨这几个点，应把 `k`、每核 rows、group 数、Sort32 repeat 长度一起纳入模型，而不是继续全局扫一个 row cap。

这个方向最多是几个百分点，因此放在 FP16 small-k 和 medium-k 覆盖之后。除非新的 Judge 评分证明 case 7 的权重异常高，否则不优先。

## 低优先级 / 暂停方向

以下方向已有足够负证据，暂不继续：

- pivot / threshold + dynamic mask compaction：真实 Judge 明显退化；
- bulk prefix pack / 单纯 barrier 数量优化：simulator 指标大降但真实 Judge 基本不动；
- Round 7 手工删 transform barrier：Bisheng auto-sync 会重新插回；
- 单纯扩大全局 batchRows：Round 8 已证明不同 shape 的收益方向不一致；
- 继续围绕 true-k extraction 做微调：Round 6 已证明元素数减少不是 Round 5 大收益的核心来源。

## 建议执行顺序

1. 先做 **FP16 vendor-perf API/layout probe**，只回答 half 能否复用当前 vendor 结构；
2. 若成立，production 化 FP16 `k<=8`，直接 Judge，重点看 case 12；
3. 再做 **keep=16** 的最小 FP32/FP16 probe，决定 medium-k 是否沿用同一架构；
4. 根据 medium-k 结果决定 large-k 是继续 prefix merge，还是只优化 generic TopK 的批处理组织；
5. 最后再回来做 3/7/9/13 的 shape-aware batch tuning。

每个阶段都要求：一个明确假设、一个短期分支、一个可复现 probe、一次真实 Judge 结论；实验结束后立即决定“合 main / 归档”，不再形成新的长期 round 链。
