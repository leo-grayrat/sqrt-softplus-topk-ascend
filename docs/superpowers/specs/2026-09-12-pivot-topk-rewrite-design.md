# Pivot Top-K Rewrite 设计

## 目标

在独立分支 `pivot-topk-rewrite` 上重写 FP32 小 k 的 Top-K 路径，不再围绕现有 `Sort32 + MrgSort` 管线做微调。目标是给中大 E 的测试点创造 2× 以上的结构性提速可能；若完整实现后 Judge 没有出现这种量级的趋势，则停止该路线，不继续做碎片化参数实验。

## 范围

第一版只替换当前 production 中 `dtype=float32 && 2<=k<=8` 的路径。`k==1` 保留 ReduceMax；FP16 和 `k>8` 保留已验证的 NORMAL TopK fallback。小 E 可以继续 fallback，重点针对 E=128..1024。

旧 `dense-repeat` production gate 在本分支撤掉，不再继续修补。相关 probe/issue 保留作为历史实验记录。

## 核心算法

每行只搬入 UB 一次，然后执行 exact pivot-filter：

1. 维护当前值域下界/上界，取 pivot。
2. 用 `Compare/Compares -> GetCmpMask` 对整行做阈值比较，并统计大于 pivot、以及必要时大于等于 pivot 的元素数。
3. 根据计数调整 pivot，目标不是精确找到第 k 大值，而是把“确定包含全局 top-k”的候选集合压到一个很小的上限（第一版设为 64）。
4. 一旦满足 `k <= candidate_count <= 64`，用用户自定义 pattern 的 `GatherMask` 把候选紧密收集到小 buffer。
5. 候选 <=32：一次 `Sort32`；候选 <=64：两次 `Sort32` + 一次很小的 `MrgSort`。最后只取前 k 个 value，再做现有 softplus+sqrt。
6. 若在固定迭代上限内无法把候选压到安全范围，或遇到大量重复值导致候选过大，直接 fallback 到现有正确路径。fallback 只保证正确性，不作为性能目标。

因为比赛输出只有排序后的 score，不输出 index，所以相等值不需要保留稳定 index 语义；但候选过滤必须保证 top-k value 一个不漏。

## 正确性条件

选择 pivot 时同时考虑 `count(>pivot)` 和 `count(>=pivot)`：

- 若 `count(>pivot) >= k`，top-k 全部位于 `>pivot` 中，可提高阈值继续缩小。
- 若 `count(>pivot) < k <= count(>=pivot)`，第 k 大值等于 pivot；若 `>=pivot` 候选数不超过上限，可直接收集该集合。
- 若 `count(>=pivot) < k`，阈值过高，需要降低。

只有在候选集合被数学上证明包含全部 top-k 时才进入最终小排序；否则继续迭代或 fallback。

## Ascend C 映射

优先利用 A2/A3 已有基础指令，不造一条新的完整排序网络：

- 比较：`Compare` / `Compares`；
- 比较结果：CmpMask + `GetCmpMask`；
- 位计数：`GetBitCount`；
- 稀疏压缩：`GatherMask` 用户自定义 pattern；
- 最终小排序：`Sort32`，必要时加一次 `MrgSort`；
- 数学变换：复用现有 `TransformSelected`。

第一版允许在候选阶段为最多 64 个值生成小 index buffer；不再为整行 E 个元素维护长期 `(value,index)` pair。

## 开发方式

不再做连续的微优化 A/B。只允许一个接口可行性门槛，然后直接完成算法：

1. 最小 CPU twin-debug probe 只验证 `Compare/GetCmpMask -> count -> GatherMask custom pattern` 的真实布局和 API 语义，不测性能。
2. probe 通过后，直接实现完整 row selector 和 fallback。
3. 一次完整 CPU correctness 回归，覆盖随机值、重复值、Inf、非 32 对齐 E，以及 k=2/4/8。
4. correctness 通过后只做一次 CANNJudge 性能判断。

## 成功/停止标准

这条路线的价值在“算法结构变化”，因此不以 5%～20% 为成功：

- 若中大测试点出现约 2× 或更强的稳定下降趋势，继续优化 pivot 迭代和 shape gate；
- 若整体仍停留在几十个百分点以内，认为 Ascend 上比较扫描/压缩成本不足以击败当前排序路径，停止该路线；
- 不在失败后继续追加 barrier、repeatTimes、batch size 等微实验来挽救它。

## 分支隔离

所有代码和 probe 只进入 `pivot-topk-rewrite`。main 保持当前已通过 Judge 的版本，直到新路径 correctness 和性能都证明值得合并。
