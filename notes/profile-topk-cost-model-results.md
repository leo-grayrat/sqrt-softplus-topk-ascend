# dav-2201 TopK primitive 成本模型：首轮结果

日期：2026-09-12

形状：`rows=1, E=1024, k=8`。对比当前 production baseline（dense-repeat dispatcher）与失败的 pivot-filter。simulator 只用于比较指令/pipe 结构，不替代 CANNJudge 实际时延。

## 1. pipe 结构

Baseline：

- VECTOR: 28,030 cycles / 4.19 us
- SCALAR: 25,274 cycles / 13.35 us
- MTE2+MTE3: 1,089 cycles / 0.58 us

Pivot：

- SCALAR: 30,116 cycles / 15.84 us
- VECTOR: 10,394 cycles / 5.61 us
- MTE2+MTE3: 1,056 cycles / 0.57 us
- ALL barrier: 204 cycles / 0.11 us

不同 pipe 的 cycle 不能直接相加后当作实际 latency；simulator 的 pipe time 也不能替代 Judge 时间。但结构变化非常清楚：pivot 把重心进一步推向 SCALAR，MTE 基本没有变化。

按 summary 中各 pipe 的 `time_us` 简单相加，baseline 约 18.12 us，pivot 约 22.13 us（+22%）。这个数字只作为 simulator 内部方向性信号，不用于预测 Judge 比例。

## 2. baseline：Sort32 / MrgSort 不是主导成本

Baseline 最显眼的 vector 指令：

- `BAR`: 25,302 cycles，127 calls
- `VBS32`: 562 cycles，1 call
- `VMS4V2`: 506 cycles，3 calls
- `MOV_UB_TO_UB`: 672 cycles，42 calls

因此：

1. `Sort32` 本身并不贵。`chunkCount=32` 的局部排序在指令层表现为一次 `VBS32`，而不是 32 个昂贵的软件 sort。
2. `MrgSort` 本身也不贵。三层 merge 只对应 3 次 `VMS4V2`，合计约 506 cycles。
3. VECTOR pipe 的绝大多数 cycle 来自 barrier：`BAR` 占 25,302 / 28,030，约 90%。
4. dense-repeat 的 `PackPrefixes` 逐 queue 做 UB->UB DataCopy 并在每次之后 `PipeBarrier<PIPE_V>()`，与大量 `BAR` / `MOV_UB_TO_UB` 一致。

这直接否定了此前两个候选判断：“MrgSort 主导”与“Sort32 主导”。继续围绕减少 Sort32 次数或减少 merge 层数做算法替换，收益上限很低。

## 3. baseline 的大头更偏 scalar / 同步 / 框架开销

Baseline SCALAR 25,274 cycles，其中：

- `STI_XN_IMM`: 8,549 cycles
- `ST_XD_XN_IMM`: 7,620 cycles
- `STP_XI_XJ_XN`: 4,899 cycles

三类 store 合计 21,068 cycles，占 SCALAR 约 83%。代码行热点还大量落在 `TPipe`、block sync、DataCopy 实现内部。

因此当前 baseline 的成本更像：大量 barrier、局部搬运和标量侧状态/流水控制，而不是 TopK 算术 primitive 本身。

## 4. pivot：真正昂贵的是 vector -> scalar 往返，不是 Compare 本身

Pivot 新增/显著放大的指令：

- `MOVEMASK`: 2,650 cycles
- `WAIT_FLAG`: 1,660 cycles（baseline 为 0）
- `VCMP`: 1,216 cycles
- `VMOVMASK_XD`: 1,024 cycles
- `SBCNT`: 128 cycles
- scalar load/store 也进一步上升

其中 `VCMP` 自身只占 pivot 总 instruction cycles 的小部分。真正的问题是 `Compare -> GetCmpMask -> PipeBarrier -> GetValue -> SetValue -> popcount -> scalar branch` 这一整条依赖链。

这次 `E=1024`，每轮有 16 个 64-element block。profile 中：

- `VCMP = 64 calls`
- `VMOVMASK_XD = 64 calls`
- `SBCNT = 64 calls`

说明该输入实际执行了 4 次整行 mask scan（16 * 4 = 64）：初始 pivot=0 后，又进行了若干轮二分 pivot。每个 block 都发生一次向量比较、mask 提取、标量读取/计数，形成不可隐藏的 vector/scalar 依赖。

这与 CANNJudge 中 pivot-filter 的集中退化方向一致：CUDA 上 ballot/popcount/register 标量逻辑很便宜的设计，映射到 dav-2201 后变成频繁的 vector/scalar crossing 和 wait。

## 5. MTE 不是当前核心问题

Baseline 与 pivot 的 MTE2+MTE3 几乎相同：

- baseline: 1,089 cycles
- pivot: 1,056 cycles

因此这轮没有证据支持“GM/UB 数据搬运是 pivot 退化主因”。布局与 MTE 不是当前首要矛盾。

## 6. 与 Judge 的关系

单行 simulator 没有复现 Judge 中 1.8x~3.4x 的完整放大比例；这一点不能强行解释。它只说明方向：pivot 的 scalar/vector 同步和 mask 读取明显增加，而 Sort32/MrgSort 并未成为主要成本。

可能的放大来源包括真实 NPU 的多 block 调度、并发/流水重叠方式、每行重复的控制开销等。当前 profile 不足以把 Judge 比例精确分解，因此不把 simulator 时间当成 Judge 时间。

## 7. 下一步设计约束

由本轮成本模型得到的约束：

1. 不再选择需要对每个 64-element block `GetCmpMask -> GetValue -> scalar count/branch` 的 selection 算法。
2. 优先保留 `Sort32` / `MrgSort` 这类硬件高级 primitive；即使表面算术工作量更多，只要能保持长向量/高级指令链，可能比反复读回标量更合适。
3. baseline 若继续优化，优先减少 per-queue barrier、UB->UB prefix packing 和 scalar/TPipe 控制，而不是继续减少 Sort32 或 MrgSort 数量。
4. dense-repeat 的 `PackPrefixes` 是一个明确可测的目标：若能用少量 bulk/strided UB copy 替代逐 queue `DataCopy + PipeBarrier`，才值得做下一轮 simulator 验证。
5. 不再救 pivot-filter；其失败属于架构映射不合适，而不是再调一两个 pivot/迭代次数即可解决的问题。

## 结论

首轮 profiling 已完成最初目的：

- `MrgSort dominates`：否；
- `Sort32 dominates`：否；
- `scalar/sync dominates pivot`：是；
- `MTE/occupancy/layout dominates`：至少从本轮 MTE 指标看否。

下一阶段应围绕“减少同步与标量控制、让 selection 保持在向量/高级 primitive 内”设计，而不是继续做 pivot-filter 或单纯减少排序/归并次数。