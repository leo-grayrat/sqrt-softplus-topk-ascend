# B=4 跨行 dense TopK profiling spike

日期：2026-09-13

## 背景

`bulk-prefix-pack` 给出了一个重要负结果：在 `rows=1,E=1024,k=8` simulator 中，单次 strided DataCopy 将 `BAR` 从 127 calls / 25,302 cycles 降到 49 calls / 6,014 cycles，VECTOR pipe 从 28,030 cycles 降到 8,143 cycles；但确认实际提交 bulk 版本后，CANNJudge 15 个测试点与旧 dense baseline 基本一致。因此不能把 simulator 中 barrier / VECTOR cycle 的账面下降直接解释成真实 NPU 墙钟收益。

随后固定 `E=1024,k=8,blockNum=1` 做 rows scaling：

- rows=1：2.89 us
- rows=2：4.52 us
- rows=4：7.80 us
- rows=8：14.34 us

总 running time 近似：

`T(rows) = 1.25 + 1.64 * rows us`

SCALAR cycles 近似：

`C_scalar(rows) = 21,324 + 3,949 * rows`

因此一次性 kernel / TPipe 固定成本存在，但不足以解释 Judge 中 6x、20x、50x 的大差距；主要问题仍在每行重复工作。

## 假设

当前 dense-repeat 虽然一个 block 可以循环处理多行，但底层 vector API 仍在 `ProcessRow()` 内逐行发射。每一行都会独立执行 ArithProgression、Sort32、prefix pack、两层 MrgSort、final merge、GatherMask 与 transform，并重复产生一批 scalar 参数准备、地址计算、flag/wait 与 API 状态开销。

本 spike 只验证：把可规则批处理的前半段跨 4 行合并，是否能显著降低同一 block 处理 4 行的 simulator running time。

## 固定形状

第一版只支持：

- FP32
- rows=4
- E=1024
- inner=1024
- k=8
- blockNum=1

不做通用 production gate，不处理 padding，不扩展 B=8。

## batch4 数据流

4 行一次搬入 UB。每行 1024 个 float，共 4096 个元素。

1. 一次 `ArithProgression<int32_t>` 生成 4096 个连续 payload index；index 只作为 `Sort32` 的伴随 payload，最终结果不使用 index。
2. 一次 `Sort32(..., repeatTimes=128)`，因为每行有 32 个 32-element chunk，4 行总共 128 repeats。
3. 用一次 strided `DataCopyParams` 将 128 个 chunk 的前 8 个 `(value,index)` pair 压成连续 128 条 prefix queue。
4. `MrgSort.repeatTimes=32`：128 条 queue -> 32 条 queue。
5. 再次 strided pack。
6. `MrgSort.repeatTimes=8`：32 条 queue -> 8 条 queue。此时布局仍按 row 保持连续，每行恰好剩 2 条 queue。
7. 第一版 final 2-way merge 仍逐行做 4 次；随后逐行 GatherMask / softplus+sqrt / 写回。这样只测前两层批量发射，不把尾层特殊布局优化混入同一实验。

## UB 粗算

按当前布局：

- input：16 KB
- index：16 KB
- Sort32 pair 输出：32 KB
- dense A：8 KB
- dense B：8 KB
- merge scratch：8 KB
- final scratch：1 KB
- output / transform scratch：< 1 KB

总计约 90 KB，明显低于 dav-2201 192 KB UB。第一版选择 B=4；不直接上 B=8。

## 第一轮 A/B 与归因问题

第一轮比较了：

- `bulk_control`：`bulk-prefix-pack` 的通用逐行 production kernel；
- `batch4`：固定 `rows=4,E=1024,k=8` 的 profiling-only batch kernel。

结果：

- bulk_control：duration 7.260 us，running 6.170 us；
- batch4：duration 5.260 us，running 4.980 us；
- running time 降低 19.3%，1.239x speedup。

结构变化：

- SCALAR cycles：32,489 -> 19,148；
- `ST_XD_XN_IMM`：11,016 -> 4,083；
- SCALAR `WAIT_FLAG`：1,310 -> 0；
- MTE2 cycles：3,482 -> 2,186；
- MTE3 cycles：2,419 -> 1,274；
- `VBS32` calls：4 -> 1，但 cycles 2,248 -> 2,194；
- `VMS4V2` calls：12 -> 6，但 cycles 2,024 -> 1,988；
- VECTOR BAR cycles 基本不变：59,542 -> 59,384。

这说明“跨行重组后标量/API 状态成本显著减少”的方向是对的，但第一轮 A/B 不能把 19.3% 全归因于 batching：`batch4` 同时把 rows/E/k/chunkCount 固定成了编译期常量，而 `bulk_control` 仍是通用 runtime kernel。大量 scalar store 的下降可能同时来自编译期专门化、循环展开与分支消除。

因此第一轮只证明“固定形状 + batching 的组合有 19.3% simulator 收益”，不能证明 batching 本身有 19.3%。

## 第二轮三组归因实验

新增 `specialized_sequential`：

- 与 batch4 相同，固定 `rows=4,E=1024,k=8,blockNum=1`；
- buffer 尺寸、k、chunkCount、merge 层数均为编译期常量；
- 明确展开固定的 `32 -> 8 -> 2 -> 1` merge 序列；
- 但仍按 `for row=0..3` 顺序执行四次单行 Sort32 / Pack / Merge / Gather / Transform，不做跨行 vector batching。

三组含义：

1. `bulk_control`：通用 + 逐行；
2. `specialized_sequential`：固定形状 + 逐行；
3. `batch4`：固定形状 + 跨行 batching。

归因：

- `bulk_control -> specialized_sequential` = 编译期专门化收益；
- `specialized_sequential -> batch4` = batching 的额外收益；
- `bulk_control -> batch4` = 两者合计。

第二轮才用于判断 batching 是否值得继续 production 化。

## 第二轮判定

以 simulator running time 为方向性指标：

- 若 `specialized_sequential -> batch4` 仍 >=20%：batching 本身为正，值得继续研究；
- 若 batching 额外收益仅 10-20%：有真实收益，但不足以解释榜首数量级差距；
- 若 batching <10%，而 `bulk_control -> specialized_sequential` 明显：第一轮收益主要来自编译期专门化，应转向 production specialization；
- 若两者都小：停止该支线。

无论第二轮结果如何，都不把 simulator 百分比直接等同 Judge 百分比。`bulk-prefix-pack` 已证明 simulator 的 barrier 记账可与真实 NPU 收益脱钩。

## 范围约束

这是 profiling-only spike。代码只放在 `profile-topk-cost-model` 分支的 `tools/` 下；不改 `submission/`，不提交 Judge。若第二轮明确识别出 specialization 或 batching 的独立正信号，再另开 production 设计。
