# Notes

## 当前入口

- 2026-10-03 最新 Judge 15/15，case 12/14 为 124.97/537.98 us；已采集两条路径的仿真时间线，原生 FP16 排序的小范围候选见 `notes/native-half-profile-2026-10-03.md`（跟踪 #47，等待 Judge）。

- 2026-10-02 flat-copy Judge 15/15，case 12/14 为 129.17/537.77 us；下一轮 keep-4 最终合并批量化见 `notes/keep4-batch-merge-2026-10-02.md`（跟踪 #47）。

- 2026-10-02 keep-12 Judge 15/15，case 14 为 558.73 us；下一轮合并候选拷贝命令的实验见 `notes/flat-prefix-copy-2026-10-02.md`（跟踪 #47）。

- 2026-10-02 FP16 keep-4 Judge 15/15 结果与 case 14 keep-12 待测候选：`notes/keep12-c14-2026-10-02.md`。keep-12 已通过本地 CPU/设备编译，等待真实 Judge，跟踪 #47。

- 当前 canonical production baseline：`main`，基于已完成本地验证和真实 CANNJudge 15/15 Pass 的 Round 6 vendor-perf 实现。
- 当前状态与测试点映射：`notes/current-status-2026-09-15.md`。
- 下一阶段优化计划：`notes/optimization-plan-2026-09-15.md`。
- FP32 small-k vendor-perf 详细进展：`notes/vendor-perf-topk-progress-2026-09-14.md`。
- 早期逐行/双行/通用 TopK 实验：`notes/perf-experiments.md`。

## 已确认原则

- `sqrt(softplus(x))` 对 logits 单调递增，因此可以先取 logits Top-K，再只计算选中的分数。
- 真实性能结论以 CANNJudge / 真实 NPU 为准；simulator 主要用于解释指令与流水成本。
- `main` 表示当前可信生产基线。实验分支完成后必须回到 main 或明确归档，不再让不断递增的 round 分支代替主线。
- 发现明确 CANN/Bisheng/API 问题时保留可复现实验和记录；已经结束的方向不继续保留为“待处理主线”。
