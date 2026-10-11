# SqrtSoftplusTopK Ascend

2026 CANN 挑战赛初赛 `SqrtSoftplusTopK` 算子的实验仓库。

仓库已经接入真实 CANNJudge starter project，并在 CANN 9.0 / dav-2201 上完成多轮 Ascend C 实现、CPU twin-debug、Bisheng 编译和真实 Judge 性能实验。

## 当前主线

`main` 是唯一生产基线。FP16 case5七叶子紧凑keep16阶段通过用户完整CANNJudge **15/15 Pass、错误占比均0.00%**，经[PR #65](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/pull/65)合入main。case5 **7.76→6.58 us，减少15.21%**，低于模型的31%–33%。[实现与全部证据](notes/fp16-compact-c5-2026-10-11.md)。

最新同次完整Judge：case7/9为 **16.47/24.39 us**，case3/13为 **115.13/132.17 us**，case6为 **10.29 us**，case12/14为 **122.26/536.85 us**，case15为 **1.65 ms**。case4增加0.09 us（2.32%）等变化全部保留，不据单次跨轮回报认定因果。[完整状态](notes/current-status-2026-10-11-c5.md)。case15 scratch仅约1.3%模型收益，已撤回并[归档](notes/fp32-large-scratch-2026-10-11.md)。

新的case6候选在`exp/fp32-compact-c6`，尚待Judge、未合main：五叶子紧凑keep16把合并产物减半，CPU真实入口双方104/104，设备编译与完整设备simulator通过，64/128行模型减少32.61%/35.44%。[实现与待测文件说明](notes/fp32-compact-c6-2026-10-12.md)。本轮[Cast](notes/fp16-padded-cast-2026-10-12.md)与[case8](notes/fp16-batch-merge-c8-2026-10-12.md)负结果已归档并撤回生产。

当前生产文件：[submission/dense_repeat_override.h](submission/dense_repeat_override.h)。这些数值来自同一次完整提交；没有拼接不同分支的单点最优结果。失败的阈值研究只保留在 debug 和 notes，没有接入生产。

- [当前状态、完整 15 点和阶段集成范围](notes/current-status-2026-10-11-c5.md)
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

现有历史实验分支保留以维持旧链接可追溯性；已合并的实验分支不再追加开发。最近已完成阶段分支为 `exp/fp16-compact-c5`，上一轮为 `exp/fp32-compact-c7-c9`；更早六条 exp 分支清单见 2026-10-04 状态文档。

## 验证原则

真实性能结论以 CANNJudge / 真实 NPU 结果为准。Simulator 主要用于解释指令和流水成本，不把 simulator 的绝对时长直接当作 Judge 性能。

README 只描述当前仓库状态；早期逐行 TopK、双行 TopK 等历史记录仍保留在 `notes/perf-experiments.md`。
