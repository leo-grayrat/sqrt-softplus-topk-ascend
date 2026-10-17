# SqrtSoftplusTopK Ascend

2026 CANN 挑战赛初赛 `SqrtSoftplusTopK` 算子的实验仓库。

仓库已经接入真实 CANNJudge starter project，并在 CANN 9.0 / dav-2201 上完成多轮 Ascend C 实现、CPU twin-debug、Bisheng 编译和真实 Judge 性能实验。

## 当前主线

**待测候选**：[C15逐行排序阈值](notes/c15-order-filter-2026-10-17.md)，分支`exp/c15-order-statistic-filter`。取消矩估计，以每行第64大组最大值构造阈值，保留精确计数和回退；均匀持续batch及32核生产模型、本地输出和设备编译通过，尚待完整Judge，main不变。

**已停止候选**：[C15矩估计筛选](notes/c15-batch-filter-2026-10-17.md)完整Judge15/15，但case15由1.02升至1.26 ms（慢23.53%），不合main；正态模型改善未兑现。

`main`更新至PR #84的case7/9/10/15合并阶段：用户完整Judge **15/15 Pass，误差全部0.00%**，四点分别为 **14.42/21.18/36.16/1020 us**，较上一轮减少11.91%/12.55%/14.37%/5.56%。case14仍为251.03 us。[完整15点、真实收益与阶段局限](notes/current-status-2026-10-17-multi.md)。

[五叶合并、镜像双值组筛选与C15复用](notes/multi-selection-2026-10-17.md)的本地模型、失败与源码审查独立保留。本轮获得改善，仍未填平前32差距；不把模型或正确性覆盖当作排名收益。

[十块精确筛选、成对搬运与缓冲复用](notes/c14-ten-blocks-2026-10-16.md)：仅FP16 E512/k10，Sort32输入192→160值、merge产物128→80对，batch16→24。4096行/32核模型减少23.47%，本次完整Judge减少23.15%，分别记录。完整形状仿真超时和CPU失败方案仍保留。[CANNBot与CANN9官方资料核对](notes/cannbot-resources-2026-10-16.md)。

此前[case13](notes/current-status-2026-10-16-c13.md)、[case3](notes/current-status-2026-10-16-c3.md)、[case15](notes/current-status-2026-10-15-c15.md)、[case14第一阶段](notes/current-status-2026-10-15-c14.md)和[case12](notes/current-status-2026-10-12-c12.md)均已集成。后续实验从新的已验证main开始。

当前生产文件：[submission/dense_repeat_override.h](submission/dense_repeat_override.h)。上方main的Judge数值来自同一次完整提交，没有拼接不同分支的单点最优；本地模型另见实验记录。旧阈值失败研究保留在debug/notes，本实验分支的批量精确筛选另行待测。

- [当前主线完整15点](notes/current-status-2026-10-17-multi.md)
- [case14第二阶段完整15点](notes/current-status-2026-10-17-c14.md)
- [上一轮case13结构性优化与完整基线](notes/current-status-2026-10-16-c13.md)
- [上一轮case3结构性优化与完整基线](notes/current-status-2026-10-16-c3.md)
- [上一轮case15结构性优化与完整基线](notes/current-status-2026-10-15-c15.md)
- [上一轮case14结构性优化与完整基线](notes/current-status-2026-10-15-c14.md)
- [上一轮case12结构性优化与完整基线](notes/current-status-2026-10-12-c12.md)
- [上一轮 case7/9与完整基线](notes/current-status-2026-10-11-c7-c9.md)
- [上一轮 case 3 连续合并与完整基线](notes/current-status-2026-10-11-c3.md)
- [上一轮 case 13 紧凑合并与完整基线](notes/current-status-2026-10-11.md)
- [上一阶段基线与分支收拢记录](notes/current-status-2026-10-04.md)
- [用户提供的结构性优化深度研究报告](notes/structural-optimization-research-2026-10-10.md)
- [FP32 scratch 复用实测：改善 2–10%，未达准入门槛](notes/fp32-scratch-reuse-2026-10-10.md)
- [报告重述与归档说明](notes/structural-optimization-research-reading-2026-10-10.md)
- [原下一阶段计划（已停止推进，保留历史）](notes/optimization-plan-2026-10-08.md)
- [values-only 网络研究：正确性通过，完整模型约慢 2.38 倍，停止](notes/values-only-c12-2026-10-08.md)
- [FP16 阶段归档](notes/fp16-optimization-archive-2026-10-03.md)
- [FP32 medium-k 实测](notes/fp32-medium-k-2026-10-04.md)

## 题目

输入 `router_logits`，形状为 `(T, E)`，逐元素计算：

```text
sqrt(softplus(x))
```

然后逐 token 取最大的 `k` 个分数。题目范围包括 FP16/FP32、`E=64..1024`、`k=1..64`。

`sqrt(softplus(x))` 严格单调递增，因此可以先在原始 logits 上完成 Top-K，再只对选中的 `T*k` 个值做变换：

```text
TopK(sqrt(softplus(x)))
=
sqrt(softplus(TopK(x)))
```

这是当前所有实现的基本数学化简。

## 仓库结构

- `submission/`：比赛提交工程与当前生产实现；
- `debug/`：CPU twin-debug / probe；
- `tools/`：结构 checker、profiling 等辅助脚本；
- `reference/`、`tests/`：PyTorch 参考实现与基础正确性测试；
- `notes/`：性能实验、负结果、源码调研与阶段总结。

## 分支约定

后续把 `main` 视为唯一 canonical baseline。新性能假设可以开短期实验分支，但实验结束后必须明确归类：

- 若形成新的可信生产基线，整理后回到 `main`；
- 若为负结果或单纯 A/B，只保留 commit / notes 作为历史证据，不继续把该分支当开发主线；
- 不再用不断递增的 `roundN` 分支替代主线集成。

每个大阶段只维护一个活动实验分支，从最新 main 创建。阶段完成后，经本地验证、用户完整 Judge 和 PR 合入 main，再开始下一阶段；不从上一条实验分支继续串接新的长期开发。

现有历史实验分支保留以维持旧链接可追溯性；已合并的实验分支不再追加开发。最近已完成阶段分支为 `exp/multi-case-selection-c7-c9-c15`，此前为 `exp/c14-exact-ten-blocks`；更早六条 exp 分支清单见 2026-10-04 状态文档。

## 验证原则

真实性能结论以 CANNJudge / 真实 NPU 结果为准。Simulator 主要用于解释指令和流水成本，不把 simulator 的绝对时长直接当作 Judge 性能。

README 只描述当前仓库状态；早期逐行 TopK、双行 TopK 等历史记录仍保留在 `notes/perf-experiments.md`。
