# SqrtSoftplusTopK Ascend

2026 CANN 挑战赛初赛 `SqrtSoftplusTopK` 算子的实验仓库。

仓库已经接入真实 CANNJudge starter project，并在 CANN 9.0 / dav-2201 上完成多轮 Ascend C 实现、CPU twin-debug、Bisheng 编译和真实 Judge 性能实验。

## 当前主线

`main` 更新至PR #74的case14精确分块筛选阶段：用户完整Judge **15/15 Pass，错误占比全部0.00%**，case14 **536.41→326.76 us，减少39.08%**。同次case12为48.76 us，case15为1.65 ms。[同次完整15点与阶段集成](notes/current-status-2026-10-15-c14.md)。

[case14实现、模型与失败记录](notes/c14-block-reduction-2026-10-12.md)：前期4096行/32核模型减少27.58%，完整Judge减少39.08%，分开报告。已集成的[case12阶段](notes/current-status-2026-10-12-c12.md)此前Judge 122.37→49.24 us，减少59.76%。下一阶段优先研究case15的结构性瓶颈。

`exp/c15-group-max-selection`已完成case15四路分组精确筛选候选：排序1024→512值，1043行/32核生产入口模型耗时减少22.85%。等待用户完整Judge，尚未合main。[实现、所有模型与复现](notes/c15-group-selection-2026-10-15.md)。

当前生产文件：[submission/dense_repeat_override.h](submission/dense_repeat_override.h)。这些数值来自同一次完整提交；没有拼接不同分支的单点最优结果。失败的阈值研究只保留在 debug 和 notes，没有接入生产。

- [当前主线完整15点](notes/current-status-2026-10-15-c14.md)
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

现有历史实验分支保留以维持旧链接可追溯性；已合并的实验分支不再追加开发。最近已完成阶段分支为 `exp/c12-batched-reduction`，上一轮为 `exp/fp32-compact-c7-c9`；更早六条 exp 分支清单见 2026-10-04 状态文档。

## 验证原则

真实性能结论以 CANNJudge / 真实 NPU 结果为准。Simulator 主要用于解释指令和流水成本，不把 simulator 的绝对时长直接当作 Judge 性能。

README 只描述当前仓库状态；早期逐行 TopK、双行 TopK 等历史记录仍保留在 `notes/perf-experiments.md`。
