# 当前生产基线与阶段收尾（2026-10-04）

本阶段将 `exp/fp32-medium-k-c6` 的整套已验证生产实现通过 PR 收拢到 main。main 上独立的六个历史文档提交已先整合，无生产代码冲突。本次收尾只更新文档与 Git 集成，不再改变用户刚验证的 kernel。

## 最新完整 Judge

同一次用户提交：15/15 Pass，全部错误占比 0.00%。这是一套完整版本，不是将不同分支的单点最快值混合成虚构基线。ms 显示精度保留原样。

| 测试点 | 当前用时 | 阶段前 Round 6 历史用时 |
|---|---|---|
| 1 | 2.45 us | 2.33 us |
| 2 | 2.90 us | 3.77 us |
| 3 | 139.69 us | 139.80 us |
| 4 | 3.72 us | 7.80 us |
| 5 | 7.50 us | 16.63 us |
| 6 | 10.19 us | 23.95 us |
| 7 | 17.63 us | 17.73 us |
| 8 | 33.46 us | 86.99 us |
| 9 | 27.52 us | 27.71 us |
| 10 | 42.30 us | 244.10 us |
| 11 | 51.94 us | 252.30 us |
| 12 | 122.17 us | 650.44 us |
| 13 | 176.78 us | 176.97 us |
| 14 | 537.48 us | 2.13 ms |
| 15 | 1.65 ms | 3.46 ms |

历史对照来自 [Round 6 记录](current-status-2026-09-15.md)，跨日期且没有重复测量分布，不据微小波动判定回退或排名。case 6 本轮直接前后对照为 23.84→10.19 us。

## 阶段内容与验证依据

- FP16 small-k、medium-k、large-k 的批量选择与变换，含 keep-4/12、scratch 复用、repeat 上限与原生 half 的限定路径。
- FP32 large-k 和 medium-k 的批量生产路径；最近 medium-k 验证 CPU 31/31、完整 submission 设备编译、7 次设备仿真输出比较，随后用户 Judge 15/15。
- 阈值原型以正常输入收益不足、含回退输入退化的负结果归档；只在 debug，不接入 submission。
- PR 收尾额外运行 Python reference 测试 19/19，并核对 submission 与用户验证版本一致。旧通用 TopK 的 CPU 检查器限制仍如实保留，不能把所有 CPU 路径统称为通过。

详见 [FP16 归档](fp16-optimization-archive-2026-10-03.md)、[large-k](fp32-large-k-c15-2026-10-04.md)、[medium-k](fp32-medium-k-2026-10-04.md)、[阈值负结果](threshold-device-probe-2026-10-04.md)。

## 分支管理

main 是后续唯一生产起点。本阶段纳入的六条实验分支：

- `exp/fp16-vendor-smallk-probe`
- `exp/fp16-vendor-mediumk-keep16`
- `exp/vendor-largek-merge32-64`
- `exp/vendor-fp16-throughput-c12-c14`
- `exp/fp32-large-k-c15`
- `exp/fp32-medium-k-c6`

这些分支保留为历史引用，不再作为活动开发主线；暂不批量删除，避免旧 Issue/文档中的文件链接失效。更早的 pivot、repeated ReduceMax、Round 7–10 等分支是独立历史实验，不因本阶段集成而自动视为通过或合入。

后续每一大阶段：从最新 main 开一条活动实验分支→同输入本地对照→用户完整 Judge→阶段 PR→合并并回 main。负结果写入记录后结束，不串出一串永久活动分支。未合入前 main 继续表示上一套已验证生产版本。
