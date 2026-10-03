# SqrtSoftplusTopK Ascend

2026 CANN 挑战赛初赛 `SqrtSoftplusTopK` 算子的实验仓库。

仓库已经接入真实 CANNJudge starter project，并在 CANN 9.0 / dav-2201 上完成多轮 Ascend C 实现、CPU twin-debug、Bisheng 编译和真实 Judge 性能实验。

## 当前主线

2026-10-03 实验分支阶段结果见 [FP16 优化总归档](notes/fp16-optimization-archive-2026-10-03.md)：五轮改动均经用户 CANNJudge 验证 15/15 Pass，最新 case 12/14 为 122.41/538.04 us。实验分支尚未合并 main；下文 main 的说明是原有基线。

`main` 只保存**已经完成本地验证并通过真实 Judge 的可信生产基线**。截至 2026-09-15，主线采用 `vendor-perf-topk-round6-extract-k` 的实现：

- FP32、`2 <= k <= 8` 使用参考 Huawei A2/A3 `MoeGatingTopKSoftmaxPerf` 执行结构的多行批处理路径；
- 非 32 对齐 `E` 在 UB 中补齐后进入同一 Sort32 / MrgSort 路径；
- Top-K 后按真实 `k` 批量抽取 values，再统一计算 `sqrt(softplus(x))`；
- 其余 dtype / k 范围继续走已验证的安全回退路径；
- 该基线真实 CANNJudge 15/15 Pass。

最新实验状态、已知测试点形状和后续方向见：

- [`notes/current-status-2026-09-15.md`](notes/current-status-2026-09-15.md)
- [`notes/optimization-plan-2026-09-15.md`](notes/optimization-plan-2026-09-15.md)
- [`notes/vendor-perf-topk-progress-2026-09-14.md`](notes/vendor-perf-topk-progress-2026-09-14.md)

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

现有历史实验分支暂时保留，避免在整理阶段破坏可追溯性。

## 验证原则

真实性能结论以 CANNJudge / 真实 NPU 结果为准。Simulator 主要用于解释指令和流水成本，不把 simulator 的绝对时长直接当作 Judge 性能。

README 只描述当前仓库状态；早期逐行 TopK、双行 TopK 等历史记录仍保留在 `notes/perf-experiments.md`。
