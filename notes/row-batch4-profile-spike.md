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

## 对照实验

不能直接把 main 的 rows=4 7.80 us 作为唯一 control，因为 batch4 会使用 strided prefix pack，而 bulk-prefix-pack 已知会让 simulator 自己变快但不改善 Judge。

因此同一脚本必须比较：

- `bulk-control`：`bulk-prefix-pack` 分支，原有逐行 `ProcessRow()`，rows=4, E=1024, k=8, blockNum=1；
- `batch4`：相同 bulk packing 语义，只把 ArithProgression / Sort32 / 前两层 MrgSort 跨 4 行批量发射。

这样可尽量把 simulator 的 packing 记账效应从 batching 收益中剥离。

## 成功标准

先要求 batch4 simulator correctness PASS。

性能只看 batch4 相对 bulk-control：

- 至少约 25% running_time 下降，才认为有继续价值；
- 若能接近 5 us 或更低，为强正结果；
- 若仍 >= bulk-control 的约 85%~90%，停止，不继续救 B=8、bank layout 或 final merge batching。

同时记录 SCALAR、VECTOR、BAR、WAIT_FLAG、VBS32、VMS4V2 的 calls/cycles，用来判断 API 发射与 per-row 控制是否真的减少。

## 范围约束

这是 profiling-only spike。代码只放在 `profile-topk-cost-model` 分支的 `tools/` 下；不改 `submission/`，不提交 Judge。若 spike 强正，再另开 production 设计。