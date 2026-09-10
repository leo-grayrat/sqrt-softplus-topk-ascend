# CANNBot Skills 深度研究与 Hierarchical TopK 优化报告

日期：2026-09-11

本报告针对 GitCode 上 `cann/cannbot`、`cann/cannbot-skills` 及其 Ascend C 开发资料做调查，并结合本仓库 `SqrtSoftplusTopK` 的当前实现与 CANNJudge 实测，整理后续优化路线。

> 资料性质说明：CANNBot Skills 适合用作 AI 友好的开发知识索引和工作流参考；涉及 CANN 9.0 / `dav-2201` 的具体 API 契约，仍以对应版本的官方文档、官方 sample 与本项目 CPU/Judge 实测为最终依据。

## 1. 最重要的结论

CANNBot 对本项目最有价值的不是“自动写算子”，而是它反复强调的一套工程闭环：

```text
API 契约确认
  ↓
CPU / Runtime 排错
  ↓
获得可信正确性基线
  ↓
一次只改一个性能机制
  ↓
真机 / Judge 对比
  ↓
有收益则保留，无收益则回退
```

这与本仓库已经形成的实践高度一致。前面的 hierarchical TopK Runtime Error 最终就是依靠 CANN 9.0 CPU twin debug 连续找到两个具体错误：

1. `TopK` 的辅助 `LocalTensor` 不能只传默认构造的空对象；
2. `Duplicate` 的子 Tensor 起点仍必须满足 32B 地址对齐，不能只看整个 TBuf 是否对齐。

修复后二级 TopK 的 FP32 单行 probe 已通过 285 组代表性参数，云端 Judge 当前版本也已经 15/15 Pass。

当前真正的第一优先级已经从“修正确性”切换为“建立新路径的盈利区间”。现阶段的 FP32 hierarchical 仍然为了排查 RE 强制 `batchRows=1`，而所有 `k<=16` 都启用新路径也没有性能模型支撑。

## 2. GitCode 资料地图

### 2.1 CANNBot Skills 主仓库

- https://gitcode.com/cann/cannbot-skills
- https://gitcode.com/cann/cannbot-skills/tree/master/docs
- https://gitcode.com/cann/cannbot-skills/tree/master/plugins-official/ops-direct-invoke

`ops-direct-invoke` 把 Ascend C 算子直调开发组织成可迭代流程，并把性能迭代单独抽成工作模块。对本项目最值得借鉴的是“实现与验收分离、实验可回退、每次只验证一个机制”。

### 2.2 Ascend C API 最佳实践

- https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-api-best-practices/SKILL.md
- https://gitcode.com/cann/cannbot-skills/tree/master/ops/ascendc-api-best-practices/references

与本题直接相关的内容包括：

- TBuf / TQue / LocalTensor 的位置与生命周期；
- DataCopy / DataCopyPad 的对齐选择；
- Sort / MrgSort 一类排序 API；
- 32B 对齐和流水同步；
- UB 空间的显式管理。

### 2.3 Crash / Runtime Debug

- https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-crash-debug/SKILL.md

其排查思路与本次实际故障吻合：Abort 优先看前面的 API checker；AIC error 重点看越界、非法访问、非对齐；AIV hang 再检查 Queue 配对和同步。

### 2.4 官方 Ascend C 文档入口

CANNBot 自己也倾向让 API 细节回到官方文档，而不是在 Skill 中硬编码全部版本细节。与本项目直接相关：

- TopK：
  https://asc.gitcode.com/api/SIMD-API/adv_api/sort_operations/TopK.html
- TopK Tiling：
  https://asc.gitcode.com/api/SIMD-API/adv_api/sort_operations/TopK-Tiling.html
- TPipe::InitBuffer：
  https://asc.gitcode.com/api/SIMD-API/basic_api/resource_management/TPipe/InitBuffer.html
- CPU twin debug：
  https://asc.gitcode.com/guide/programming_guide/debug_and_tuning/functional_debug/cpu_twin_debug.html

仓库检索中没有发现一个可直接替代官方 TopK 文档的独立 TopK Skill。因此 `TOPK_NSMALL`、`TopKInfo`、`GetTopKMaxMinTmpSize` 等具体契约继续以 CANN 9.0 官方资料为准。

## 3. TopK / NSMALL 对当前设计的约束

### 3.1 A2 上的可用路径

当前比赛目标为 `dav-2201`。CANN 9.0 TopK 在该代设备上支持 `half` / `float`，可使用 NORMAL 与 NSMALL。NSMALL 是针对每行 `inner=32` 的 Small 模式；A2 的 TopK 仍属于 merge-sort 路线。

`RADIX_SELECT` 虽然存在于较新平台的 TopK 算法选项中，但不是 `dav-2201` 上可直接采用的优化路径，因此不列入当前方案。

### 3.2 当前 hierarchical 的数学结构

当前实现：

```text
原始一行 inner
  ↓
切成 inner/32 个 chunk
  ↓
每个 chunk 用 TOPK_NSMALL 保留局部 k
  ↓
candidateCount = chunkCount × k
  ↓
补齐 candidateInner = AlignUp(candidateCount, 32)
  ↓
TOPK_NORMAL 取最终 k
```

这个算法在数学上成立：如果一个元素连自己所在 32 元素块的局部 top-k 都进不去，则同块至少已有 k 个元素不小于它，它不可能进入全局 top-k。

第一阶段的理论候选比例为：

```text
candidateCount / inner = k / 32
```

但真正决定第二阶段 NORMAL 代价的是 `candidateInner`，而不是 `candidateCount`。

例如：

| inner | k=2 | k=8 | k=16 |
|---:|---:|---:|---:|
| 64 | 50% | 50% | 50% |
| 128 | 25% | 25% | 50% |
| 256 | 12.5% | 25% | 50% |
| 512 | 6.25% | 25% | 50% |
| 1024 | 6.25% | 25% | 50% |

所以 `k=16` 时，第二阶段仍要处理原始长度约一半的数据，却额外付出了完整的第一阶段 NSMALL、候选写入和同步成本。由此可见，“所有 `k<=16` 一律走 hierarchical”只是一个正确性上可行的范围，不是性能上合理的范围。

## 4. 已经得到的 API / 调试经验

### 4.1 LocalTensor 有效性

`isInitIndex=false` 的含义是“不需要给 `srcIndexLocal` 填有效索引数据”，不等于“可以传一个没有 backing buffer / position 的空 LocalTensor”。CPU checker 已经实际证明这种区别。

当前 hierarchical 显式给 `srcIndexLocal`、`finishLocal` 提供 VECCALC backing buffer，是正确性保险；之后若压缩空间，只能把“最小 backing size”当成独立实验，不能凭直觉删除。

### 4.2 32B 对齐

`InitBuffer` 对整个 Buffer 的对齐，并不保证 `tensor[offset]` 仍然对齐。

本项目已经遇到：

```text
E=64, k=2
candidateCount=4
4 × sizeof(float)=16B
```

若从 `candidateLocal[candidateOffset + candidateCount]` 开始 `Duplicate`，地址相对基址偏移 16B，CPU checker 会直接报目标地址不满足 32B 对齐。

当前做法改成从对齐的候选区起点整段预填 `-inf`，再让 NSMALL 覆盖有效候选，已经通过 CPU 与 Judge 验证。

### 4.3 TopK tiling

开发者应该控制的是：

```text
inner / outter / k / dtype / mode / isInitIndex / isLargest
```

`TopkTiling` 内部的 repeat、mask、offset 等字段应由 `TopKTilingFunc` 生成，而不应自己推导并硬写。

临时空间通过 `GetTopKMaxMinTmpSize` 获取合法区间。当前取 `minTmp` 并在两阶段之间复用 `max(smallTmp, finalTmp)` 是正确性优先且节省 UB 的做法。

是否“给更大的 tmp 会更快”没有文档保证，只能以后单独 A/B。

### 4.4 CPU debug 的定位

CPU twin debug 是正确性 / API-contract gate，不是性能模拟器。当前工程的价值是直接 include `submission/kernel.asc`，避免测试器和实际 kernel 漂移。

正确解释日志：

```text
没有 RUN
→ host / tiling / 参数准备问题

有 RUN + checker ERROR / SIGABRT
→ kernel API 契约、地址、Buffer 问题

kernel 正常退出 + MISMATCH
→ 算法 / 精度问题
```

`addr2line: DWARF ...` 在此前日志中只是回溯符号化噪声；真正根因仍看其后的 CANN checker 第一条 ERROR。

## 5. 当前性能诊断

当前云端 15 case 已经全部正确，但 FP32 hierarchical 分支仍是诊断配置：

```cpp
const int32_t hierarchicalBatchRows = 1;
```

而旧 NORMAL 路径在 T 足够大时使用 `batchRows=2`。历史实测已经证明双行批处理在中大 shape 上通常能回收约 10%～19%，且 hierarchical 自身的 case 3 从 batch2 约 337us 退回 batch1 约 385us，也已经证明这不是无关紧要的差异。

当前 hierarchical kernel 还有两类结构性额外成本：

1. 每个 slot 都整段 `Duplicate(candidateInner)`，即使 `candidateCount==candidateInner` 根本没有 padding；
2. batch2 时第一阶段依旧会对两个 slot 分别调用两次 NSMALL，各自带 barrier，而最终 NORMAL 已经能够用 `outter=2` 一次处理两行。

长期最重要的瓶颈仍是高阶 TopK 本身会管理 values + indexes，而本题最终完全不需要索引。若依靠官方高阶 TopK 无法接近前排，最终很可能需要研究基于 `Sort32 / MrgSort / GatherMask` 的 values-only / partial-selection 路线。

## 6. 后续优化路线

### P0-1：先给 CPU probe 增加 FP32 batchRows=2

目的：恢复正式 batch2 前，先让 CPU checker 和 golden 覆盖真实的两行布局。

要求：

- 输入、输出各两行；
- final TopK tiling 使用 `outter=2`；
- hierarchical kernel 真实传 `rows=2, batchRows=2`；
- 两行使用不同数据并分别与 host golden 比较；
- quick 与 full 两套边界扫描都可以显式跑 batch2。

这是当前下一步。

### P0-2：恢复正式 FP32 hierarchical batchRows=2

只有 P0-1 的 quick / full 均通过后再修改正式 dispatch。

FP16 继续走旧 NORMAL，不与这一变量混合。

提交 Judge 后完整记录 15 case，作为新的 hierarchical 性能基准。

### P0-3：建立保守盈利 Gate

第一版只作为保守起点：

```cpp
useHierarchical =
    dtype == FP32 &&
    inner >= 256 &&
    k <= 8 &&
    candidateInner * 4 <= inner;
```

这不是最终阈值，而是要求第二阶段至少压缩到原始输入 25% 的保守选择。之后根据 Judge A/B 向外扩展，而不是先把所有小 k 都压在新算法上。

### P1-1：无 padding 时跳过候选 Duplicate

```cpp
if (candidateCount_ < candidateInner_) {
    Duplicate(candidateLocal[candidateOffset], -inf, candidateInner_);
    PipeBarrier<PIPE_V>();
}
```

注意仍必须从对齐的 `candidateOffset` 开始，不能恢复成从尾部未对齐位置填充。

### P1-2：自然对齐候选时合并两行 NSMALL

当 `batchRows=2` 且 `candidateCount==candidateInner` 时，研究：

```text
small.outter = chunkCount × 2
```

一次 NSMALL 直接生成：

```text
row0 candidates | row1 candidates
```

因为没有行间 padding，布局可以直接交给最终 `outter=2` NORMAL。

若 `candidateCount != candidateInner`，紧密输出和最终 stride 不一致，额外重排很可能抵消收益，所以首版不要强行覆盖。

### P1-3：TopK tmp-size A/B

只在 UB 账本确认有余量后做：

```text
min tmp
中间值
可用上限
```

small / final 阶段分开测试。CPU 只验证合法性，性能结论必须来自 Judge/NPU。

### P2：压缩辅助 Buffer

`srcIndexBuffer_` 当前按最坏需求分配。后续可通过 CPU checker 单独寻找最小合法 backing size，为更大 batch、更多 tmp 或双缓冲腾 UB。

这不是优先性能项，不应在当前阶段先做。

### FP16：独立优化线

FP16 不跟随 FP32 自动开启 hierarchical。建议顺序：

```text
FP16 batch1 CPU probe
→ FP16 batch2 CPU probe
→ conservative gate
→ Judge A/B
→ 只保留真实盈利区
```

### 长期：values-only TopK

若上述高阶 API 调度和 shape gate 仍无法把大 case 拉近前排，则进入：

```text
Sort32 / MrgSort / GatherMask
或其他 A2 上可用的小 k selection
```

目标是避免高阶 TopK 为索引维护付出本题不需要的成本。该路线潜在收益最高，但实现和正确性风险也最高，必须保留现有高阶 TopK 作为 fallback。

## 7. 建议的 shape dispatch 方向

不要只按 `k` 分支，最终应该显式考虑：

```text
dtype
inner = AlignUp(E,32)
k
candidateInner
rows / coreNum
```

短期用规则 gate，长期把 Judge / NPU 数据积累成小型策略表。未覆盖的 shape 永远回退到已经 15/15 Pass 的 NORMAL 路径。

## 8. 实验纪律

后续每次完整 Judge 都只改变一个机制：

```text
A = 旧 NORMAL 控制版
B = hierarchical batch=2
C = B + 保守 Gate
D = C + skip Duplicate
E = D + aligned batch-NSMALL
```

判断顺序：

1. 先要求 15/15 correctness；
2. 再比较本项目上一版本的同 case 时间；
3. 前 3 / 前 10 用于判断现实目标；
4. TBest 只作为历史单点下界，不直接当正常整体目标。

CPU probe 只证明 CPU checker / golden 层面；本地 Bisheng 编译、CPU 运行、云 Judge correctness、Judge 性能是不同验证层级，记录时不得混写。

## 9. 当前下一步

按收益、风险和验证成本排序，立即执行：

1. 扩展 `debug/cpu_topk_probe` 支持 FP32 `batchRows=2`；
2. 用户本地跑 batch2 quick，然后 batch2 full；
3. 两者全过后，恢复正式 FP32 hierarchical `batchRows=2`；
4. 完整 Judge 15 case；
5. 再用保守 Gate 与全覆盖 hierarchical 做一次单变量 A/B。

在这之前不重新开启 FP16 hierarchical，也不开始 values-only 重写。