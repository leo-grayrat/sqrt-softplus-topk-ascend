# Notes

## 当前入口

- **2026-10-11 紧凑合并候选待 Judge**：[FP32 k=8 / inner=224](notes/fp32-compact-merge-2026-10-11.md)。CPU 38/38，完整设备编译通过；64/128 行模型相对 main 改善 23.13%/27.15%，相对 scratch-only 改善 21.32%/24.81%；单行慢 0.23 us。新生产代码只在实验分支，待完整 15 点 Judge 后阶段 PR 合 main。跟踪 #57。

- **2026-10-10 FP32 scratch 复用已实现测量**：[完整记录](notes/fp32-scratch-reuse-2026-10-10.md)。基线/候选 CPU 各 38/38，候选完整设备编译通过；E210 改善 2.3–3.1%、E888 约 10%，未达 20% 门槛。保留可复现研究补丁，生产代码恢复 main；无需新 Judge。跟踪 #55。本次是报告第一项，未实施第二项 merge tree 或 #52。

- **2026-10-10 用户研究报告归档**：[完整原文](notes/structural-optimization-research-2026-10-10.md) / [意思重述与来源说明](notes/structural-optimization-research-reading-2026-10-10.md)。按用户要求停止推进原下一阶段计划，本次只存档和重述，尚未实施报告中的 FP32 scratch 复用等建议。

- **2026-10-08 values-only 阶段结束**：[实现与负结果](notes/values-only-c12-2026-10-08.md)。三组 CPU 各 33/33、设备构建通过；62/124 行完整模型分别 24.96/44.25 us，main 为 10.51/18.62 us。停止该网络，生产目录不变，无需新 Judge。[下一阶段计划](notes/optimization-plan-2026-10-08.md)。跟踪 #51；极大输入变换边界另见 #52。

- **2026-10-04 阶段集成**：以 [main 当前状态](notes/current-status-2026-10-04.md) 和 [下一阶段计划](notes/optimization-plan-2026-10-04.md) 为准。以下为历史实验时间线，其中“待测”“尚未合并”等仅描述当时状态。

## 历史实验时间线

- **2026-10-04 FP32 medium-k Judge 通过**：[跨行 keep-16 候选](notes/fp32-medium-k-2026-10-04.md)。新生产路径 CPU 31/31，设备编译通过；32/64 行同输入仿真耗时减少 48%/51%。Judge 15/15，case 6 从 23.84 降至 10.19 us（减少 57.3%）；分支 `exp/fp32-medium-k-c6`，#49 已验证。

- **2026-10-04 阈值完整原型停止**：[设备原型与同输入 A/B](notes/threshold-device-probe-2026-10-04.md)。CPU 正确性通过，64 行正常输入仿真改善 11.3%，未达到预设 20% 门槛；保留代码和复现脚本，不接入 submission。跟踪 #48。

- **2026-10-04 结构性优化准入实验**：[阈值筛选合同、历史负结果与停止标准](notes/threshold-selection-plan-2026-10-04.md)。主机模型两组各 3025 行通过；淘汰 256 采样方案，512/256 的后续设备原型结果见上，不是新的 Judge 候选。跟踪 #48。

- **2026-10-04 case 15 Judge 通过**：[FP32 large-k 批量拷贝与最终合并](notes/fp32-large-k-c15-2026-10-04.md)。生产路径 CPU 14/14、设备编译及两组仿真通过；独立分支 `exp/fp32-large-k-c15` 已获 Judge 15/15，case 15 从 1.69 降至 1.65 ms（单次回报），跟踪 #48。

- **2026-10-03 阶段总归档**：[FP16 case 12/14 完整优化记录](notes/fp16-optimization-archive-2026-10-03.md)。最新实验分支 Judge 15/15，case 12/14 为 **122.41/538.04 us**；原生 FP16 已过 Judge。下列日期条目保留历史过程，其“待测”不表示当前仍未验证。尚未合并 main，#47 保持开放。

- 2026-10-03 最新 Judge 15/15，case 12/14 为 124.97/537.98 us；已采集两条路径的仿真时间线，原生 FP16 排序的小范围候选见 `notes/native-half-profile-2026-10-03.md`（跟踪 #47，等待 Judge）。

- 2026-10-02 flat-copy Judge 15/15，case 12/14 为 129.17/537.77 us；下一轮 keep-4 最终合并批量化见 `notes/keep4-batch-merge-2026-10-02.md`（跟踪 #47）。

- 2026-10-02 keep-12 Judge 15/15，case 14 为 558.73 us；下一轮合并候选拷贝命令的实验见 `notes/flat-prefix-copy-2026-10-02.md`（跟踪 #47）。

- 2026-10-02 FP16 keep-4 Judge 15/15 结果与 case 14 keep-12 待测候选：`notes/keep12-c14-2026-10-02.md`。keep-12 已通过本地 CPU/设备编译，等待真实 Judge，跟踪 #47。

- Round 6 是历史基线，当前生产基线与完整 Judge 结果见 `notes/current-status-2026-10-04.md`。
- 当前状态与测试点映射：`notes/current-status-2026-09-15.md`。
- 2026-09-15 历史计划：`notes/optimization-plan-2026-09-15.md`；当前计划见顶部入口。
- FP32 small-k vendor-perf 详细进展：`notes/vendor-perf-topk-progress-2026-09-14.md`。
- 早期逐行/双行/通用 TopK 实验：`notes/perf-experiments.md`。

## 已确认原则

- `sqrt(softplus(x))` 对 logits 单调递增，因此可以先取 logits Top-K，再只计算选中的分数。
- 真实性能结论以 CANNJudge / 真实 NPU 为准；simulator 主要用于解释指令与流水成本。
- `main` 表示当前可信生产基线。实验分支完成后必须回到 main 或明确归档，不再让不断递增的 round 分支代替主线。
- 发现明确 CANN/Bisheng/API 问题时保留可复现实验和记录；已经结束的方向不继续保留为“待处理主线”。
