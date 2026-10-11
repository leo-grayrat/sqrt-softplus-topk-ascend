# FP32 k=8、inner=224 紧凑合并：Judge 通过（#57）

分支 `exp/fp32-k8-compact-merge` 从 main 开始，实施用户报告第二项，并组合上一轮已验证正确的 scratch 复用。用户已回报完整 Judge 15/15 Pass、全部错误占比 0.00%；case 13 从 176.78 降至 132.12 us（耗时减少 25.26%，约 1.338 倍加速）。通过 [PR #58](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/pull/58) 收拢到 main。生产修改仅为 `submission/dense_repeat_override.h`；kernel.asc 不变。

完整原始回报和本阶段集成范围见 [当前状态](current-status-2026-10-11.md)。本轮其他点没有明显退化；这是单次整套回报与上一套基线的比较，没有重复测量分布。

## 改变了什么

作用范围为 FP32、k=8、padded inner=224（E=193..224），主要对应 case 13。case 3 的 inner=896 不启用本轮路径，也不携带上一轮单独约 10% 收益的 scratch 实验。

通用路径给 7 个真实 Sort32 叶子安排 16 个 top-8 槽位，第一层合并 4 组，第二层合并 32 个候选。本轮改为：

1. 7 个真实叶子加一个 -inf 槽，共 8 个 top-8 槽位，第一层只合并两组。填充槽不会丢失真实重复值。
2. 两份各 32 pair 的结果用一次跨列表 strided DataCopy 提取各自 top-8，替代原先 4 次逐列表 prefix 拷贝。
3. 两份 top-8 拼接天然可拆为 4 份有序 top-4，用已有 MrgSort 四路 repeat 接口合并所有行，最终每行输出 16 pair。GatherMask 行步长同步从 64 float 改为 32 float。
4. pairB 排序前承载 padded input 和零索引，排序后恢复 merge scratch；每批重置索引。最大 batch 从 16 扩为 32，Sort32 最多 224 repeats。显式 UB 最大分配 172064 B。

每行第一层/最后一层合并产物合计从 160 pair 减为 80 pair。合并层数仍为两层，仍保留必要的 prefix pack，因此不称为“零拷贝”或“删除整层合并”。四路 split-list repeat 用法已在现有生产 FP16 keep-4 / FP32 large-k 路径使用，并核对本机 CANN 9.0 接口。

## 验证

- 新候选真实 `run_kernel` 入口 CPU **38/38**。覆盖 batch 边界、填充范围边界、邻近未修改 gate、k=7、多核与尾批次，以及全 -inf、重复值、极值集中和大幅度输入。沿用原 FP32 完整变换容差；没有额外声称 raw top-k 逐位检查。
- main、scratch-only、新候选分别重新编译 simulator，并使用同源 probe、同输入、同核数测量。
- 新候选完整 submission 的 dav-2201 编译通过。
- 原有 vendor/FP16 结构 checker 通过；vendor checker 更新为同时识别原布局和显式受限的 alias/compact 布局。它仅是接线检查，正确性结论来自生产入口 CPU 和 simulator。
- 257 行、8 核（32 行 batch 加最后 1 行尾批次）的新候选 simulator 输出检查通过。本轮共 10 次 simulator 运行全部通过输出比较。
- 基线源码核对与 main 一致，新候选源码核对与实际构建副本一致；复现脚本语法检查通过。

## 新鲜完整路径对照

2026-10-11，Ascend910B1 simulator，E=210、k=8、单核。以下为单次 `duration_time(us)`，包含输入、选择、变换和写回，不是 Judge/NPU 性能，也不代表统计分布。

| 行数 | main | 仅 scratch 复用 | scratch + 紧凑合并 | 相对 main 耗时减少 | 相对 scratch 耗时减少 |
|---:|---:|---:|---:|---:|---:|
| 64 | 16.56 | 16.18 | 12.73 | 23.13% | 21.32% |
| 128 | 30.46 | 29.51 | 22.19 | 27.15% | 24.81% |
| 1 | 3.77 | 3.75 | 4.00 | -6.10% | -6.67% |

两组长 batch 均达到已有 20% 模型门槛，相对只复用 scratch 也有独立增益。单行慢 0.23 us，保留为已知短输入代价，不能声称所有规模都变快。随后完整 15 点 Judge 已验证正确性，并确认 case 13 的实际收益；该收益不能外推为所有输入规模的提升。

## 指令佐证

| 64 行指令调用数 | main | scratch-only | 新候选 |
|---|---:|---:|---:|
| VBS32 | 4 | 2 | 2 |
| VMS4V2 | 8 | 4 | 4 |
| MOV_UB_TO_UB | 48 | 24 | 18 |
| BAR | 132 | 70 | 64 |

scratch-only 与新候选的排序/合并调用数相同，但每次合并处理的候选减少；不能仅看调用数判断工作量。UB 搬运调用进一步减少 25%。BAR 是模型总计数，不将各条指令的耗时相加当作 kernel wall time。

## 复现与交付

- [复现脚本](../tools/run_fp32_compact_merge_probe.sh)：加载 CANN 环境后分别以 `baseline`、`scratch`、`candidate` 模式运行，三个模式使用不同绝对构建目录。baseline 固定为本阶段开始的 main；scratch 在该基线应用归档补丁；candidate 使用本分支生产源码。
- [生产入口 probe](../debug/cpu_topk_probe/topk_fp32_scratch_production_probe.asc)。
- [构建/CPU/仿真日志与指令计数](fp32-compact-evidence-2026-10-11)，仅去除日志行尾空格。

用户完整 Judge 已完成，本阶段经 PR #58 合入 main，实验分支仅保留历史引用。归档收尾不改变已验证生产源码。后续阶段从 main 开始；#52 数值边界不在本次修复范围。
