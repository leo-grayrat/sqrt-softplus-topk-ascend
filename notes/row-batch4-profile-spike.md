# B=4 跨行 dense TopK profiling spike

日期：2026-09-13

## 背景

`bulk-prefix-pack` 给出了一个重要负结果：在 `rows=1,E=1024,k=8` simulator 中，单次 strided DataCopy 将 `BAR` 从 127 calls / 25,302 cycles 降到 49 calls / 6,014 cycles，VECTOR pipe 从 28,030 cycles 降到 8,143 cycles；但确认实际提交 bulk 版本后，CANNJudge 15 个测试点与旧 dense baseline 基本一致。因此不能把 simulator 中 barrier / VECTOR cycle 的账面下降直接解释成真实 NPU 墙钟收益。

随后固定 `E=1024,k=8,blockNum=1` 做 rows scaling：

- rows=1：2.89 us
- rows=2：4.52 us
- rows=4：7.80 us
- rows=8：14.34 us

总 running time 近似：`T(rows) = 1.25 + 1.64 * rows us`。
SCALAR cycles 近似：`C_scalar(rows) = 21,324 + 3,949 * rows`。

因此一次性 kernel / TPipe 固定成本存在，但不足以解释 Judge 中 6x、20x、50x 的大差距；主要问题仍在每行重复工作。

## 假设

当前 dense-repeat 虽然一个 block 可以循环处理多行，但底层 vector API 仍在 `ProcessRow()` 内逐行发射。每一行都会独立执行 ArithProgression、Sort32、prefix pack、两层 MrgSort、final merge、GatherMask 与 transform，并重复产生一批 scalar 参数准备、地址计算、flag/wait 与 API 状态开销。

本 spike 只验证：把可规则批处理的前半段跨 4 行合并，是否能显著降低同一 block 处理 4 行的 simulator running time。

## 固定形状

第一版只支持 FP32 / rows=4 / E=1024 / inner=1024 / k=8 / blockNum=1，不做通用 production gate，不处理 padding，不扩展 B=8。

## batch4 数据流

4 行一次搬入 UB。每行 1024 个 float，共 4096 个元素。

1. 一次 `ArithProgression<int32_t>` 生成 4096 个连续 payload index；
2. 一次 `Sort32(..., repeatTimes=128)`；
3. 一次 strided `DataCopyParams` 将 128 个 chunk 的前 8 个 `(value,index)` pair 压成连续 128 条 prefix queue；
4. `MrgSort.repeatTimes=32`：128 -> 32 queues；
5. 再次 strided pack；
6. `MrgSort.repeatTimes=8`：32 -> 8 queues，此时每行剩 2 条；
7. final 2-way merge / GatherMask / softplus+sqrt / 写回仍逐行做。

UB 粗算约 90 KB，低于 dav-2201 192 KB UB。

## 第一轮 A/B

第一轮比较：

- `bulk_control`：通用逐行 production kernel；
- `batch4`：固定 `4x1024x8` 的 profiling-only batch kernel。

结果：

- bulk_control：running 6.170 us；
- batch4：running 4.980 us；
- 合计下降 19.3%，1.239x。

结构变化：SCALAR cycles 32,489 -> 19,148；`ST_XD_XN_IMM` 11,016 -> 4,083；SCALAR `WAIT_FLAG` 1,310 -> 0；MTE2 3,482 -> 2,186；MTE3 2,419 -> 1,274。`VBS32`/`VMS4V2` calls 显著减少，但各自总 cycles 基本不变；VECTOR BAR cycles 59,542 -> 59,384，几乎不变。

第一轮存在混杂：batch4 同时做了固定形状专门化，因此不能把 19.3% 全归因 batching。

## 第二轮三组归因

新增 `specialized_sequential`：固定 `rows=4,E=1024,k=8,blockNum=1`，merge 路径和 buffer 尺寸全部编译期确定，但仍逐行执行四次单行 Sort32 / Pack / Merge / Gather / Transform。

三组：

1. `bulk_control`：通用 + 逐行；
2. `specialized_sequential`：固定形状 + 逐行；
3. `batch4`：固定形状 + 跨行 batching。

### 实测

| variant | duration_us | running_time_us |
| --- | ---: | ---: |
| bulk_control | 7.260 | 6.170 |
| specialized_sequential | 6.490 | 6.020 |
| batch4 | 5.260 | 4.980 |

归因：

- bulk_control -> specialized_sequential：1.025x，**2.4%**；
- specialized_sequential -> batch4：1.209x，**17.3%**；
- bulk_control -> batch4：1.239x，**19.3%**。

因此第一轮收益主要来自 batching，而不是固定形状专门化。

### 指令结构

specialized_sequential -> batch4：

- SCALAR cycles：27,086 -> 19,148；
- SCALAR calls：2,340 -> 1,507；
- `ST_XD_XN_IMM`：7,342 -> 4,083；
- SCALAR `WAIT_FLAG`：1,387 -> 0；
- `LD_XD_XN_IMM`：1,796 -> 1,278；
- MTE2：3,502 -> 2,186 cycles；
- MTE3：2,352 -> 1,274 cycles；
- `VBS32`：4 calls / 2,248 cycles -> 1 call / 2,194 cycles；
- `VMS4V2`：12 calls / 2,024 cycles -> 6 calls / 1,988 cycles；
- VECTOR BAR：51,719 -> 59,384 cycles，未下降。

这说明收益主要来自减少逐行 API 发射、标量参数/地址/状态操作和 MTE flag/wait，而不是减少 Sort32/MrgSort 算术本身，也不是减少 VECTOR BAR。

## 当前结论

1. 固定 E/k/rows 的编译期专门化单独只有 2.4%，不值得作为主方向。
2. 跨行 batching 本身在 `E=1024,k=8` 上有真实 simulator 正信号，约 17.3%，但达不到预设 25% 强正阈值。
3. 该收益不足以解释榜首数量级差距，因此不能把 B=4 production 化当成主线终点。
4. 但 production 的 FP32 `k<=8` 当前完全按行处理；对于小 E、大 rows，每行真实 vector 工作更少，而同类逐行控制成本仍存在，batching 的相对收益有可能显著放大。相比继续优化 E=1024 的 final merge，这一假设的信息增益更高。
5. 下一轮若继续本支线，应优先测小 E（如 E=64/128/256）的大 batch batching scaling，而不是 B=8 或 bank-layout 微调。

## 范围约束

当前仍为 profiling-only。正式 submission 未改。只有在小 E 上出现明显更强的 batching 收益，并且不会因减少 block 数而损失并行度时，才值得设计 production gate（例如仅在 rows 足够大、groupCount 仍能填满所有可用 vector core 时启用）。
