# FP16 values-only 跨行网络研究（#51）

## 假设和范围

从 2026-10-04 合入 main 的完整 Judge 15/15 基线出发，研究 E=256、k=2..4 的固定控制流选择网络。k=1 已有独立 max 路径，本实验不覆盖。当前 case 12 Judge 基线是 122.17 us。

原型位于 `debug/cpu_topk_probe/values_only_research_kernel.h`，没有修改 `submission/`。不降低输入精度：half 转 float 精确表示后进行比较。保留 main 的 batch 策略（最多 31 行）和双缓冲预取，比较时输入、核数和完整输出检查一致。

## 数据布局与完整成本

第一层把相邻两个值拆成偶/奇位置，生成降序的两个 rank plane；第二层用四路 bitonic merge 得到四个降序 plane。之后每层把相邻子树的 top-4 反向配对，保留上半部，再用两级比较恢复降序。子树宽度依次为 8、16、32、64、128、256。比较逐 plane 跨所有行执行，没有逐行 argmax 或候选计数回读。重复值按出现次数保留。

每个 batch 有 54 次拆分 Gather、1 次最终输出 Gather、82 次 Min/Max。还包括一次输入 half->float、索引初始化、尾行填充、已有 stable softplus+sqrt、最终 float->half 和 GM 写回。尾行使用 -inf 初始化，所有 rank plane 按 32B 对齐。

这是研究模型，不能用“去掉 index payload”推断吞吐提高。普通 Gather 和比较网络可能比专用 Sort32/MrgSort 更贵。预先规定两组连续 batch 的完整模型时间至少改善 20%，才进入生产接入和全套 Judge。

## 验证方法

- `tools/check_values_only_network.py`：1,225 组含重复值和 -inf 的穷举四路合并，以及 1,000 个 256 列完整行，全部通过。
- 三个同源 probe：`topk_values_only_probe` 执行候选完整路径；`topk_values_only_baseline_probe` 调用 main 的生产 dispatch；`topk_values_only_raw_probe` 直接输出选中的原始值并精确比较。
- 每个 probe 33 个形状：k=2/3/4，单核行数 1/3/30/31/32/62/63/124，多核行数 17/249/497。覆盖完整/尾 batch、单段极值、末尾极值、重复值、全负无穷、微小值。raw probe 另含 FP16 最大有限值。
- `tools/run_values_only_probe.sh /absolute/build-directory`：在已加载 CANN 9.0 环境的 Linux/WSL 中构建 CPU/sim，运行检查，再比较单核 62/124 行（2/4 batch）的完整路径。模型结果不等于真实 NPU 或 Judge。

## 数值边界独立记录（#52）

最初完整 probe 加入 top value=65504 后，候选和 main 均输出 256；host 精确变换为 255.937485、host half 为 255.875。误差略超原 probe 的 0.5 ULP + 2e-6 门槛。这是双方共同的已复现边界，未放宽容差、未宣称该输入的完整变换通过。最大有限值继续由 raw probe 精确检查选择；完整变换 probe 保留原范围以及 -inf/微小值。不能据此认定当前已通过 Judge 的主线出现新回归。跟踪 [#52](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/52)。

## 实测结果

CPU：候选完整路径 33/33、main 生产 dispatch 33/33、raw 精确选择 33/33。候选与 main 的 sim 设备编译均通过；没有真实 NPU/Judge 新结果。

CPU 检查期间，31 行的最终 Gather 曾在模型读出阶段崩溃。将输出索引缓冲按 64 个 uint32 对齐，并把非活动 lane 的索引显式初始化为 0 后，上述 99 个检查通过；保留这种完整初始化，不能依赖被掩蔽 lane 中的未初始化索引。这里记录的是本机 CANN 9 CPU 模型的实际复现与修复，不推广为所有 Gather API 的额外合同。

Ascend910B1 simulator，单核、E=256、k=3，比较 `duration_time(us)`；四次运行都通过输出比较：

| 行数 | 连续 batch | main | values-only | 候选/main |
| --- | --- | --- | --- | --- |
| 62 | 2 | 10.51 us | 24.96 us | 2.375x |
| 124 | 4 | 18.62 us | 44.25 us | 2.377x |

结论：两组完整成本均明显退化，未达到预设改善 20% 门槛。原型正确，但“去索引的普通 Gather + Min/Max 网络替代专用排序”在这个实现上不成立。停止该网络，不扩到 case 14、不通过扫 batch/阈值继续这个方向，不接入生产、不申请新 Judge。原始构建、CPU、模型日志保存在 [evidence](values-only-evidence-2026-10-08)。

本阶段以研究代码和负结果 PR 归档到 main，生产目录与上一套 15/15 Judge 基线完全一致。#51 保留记录；数值边界 #52 仍待独立验证。下一研究线见 [下一阶段](optimization-plan-2026-10-08.md)。

归档日志仅去除行尾空格，结果内容未改写。
