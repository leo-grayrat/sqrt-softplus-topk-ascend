# Case 15 整行阈值筛选设备原型

跟踪 [#48](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/48)，承接[准入计划](threshold-selection-plan-2026-10-04.md)。生产 submission 保留 15/15 Pass、case 15 为 1.65 ms 的版本。

## 实现范围

研究代码在 `debug/cpu_topk_probe/threshold_research_kernel.h`，由当前 large-k 类快照扩展，不接入 production dispatch。基线目标直接调用 production 类；两个完整路径目标使用同一 host harness、相同输入、相同核数和数值参考。

每行固定抽取偶数位置的 512 个值，精确取样本 Top-64 的末位作阈值。一次整行 CompareScalar 生成 1024 位掩码，GatherMask 整体压缩，读取一次候选计数。压缩缓冲可容纳全部 1024 个值，先补 -inf；计数在 64..256 时精确排序前 256 个槽位，否则走完整 1024 Top-K。阈值读取仍需向量到标量同步，整体压缩并不等于消除同步。

使用原类输入双缓冲、按批输入/输出及分数变换；选择阶段因每行阈值和候选数不同逐行处理，并保存各行结果。原型只支持 E=1024、k=64、FP32，NaN 没有获得完整合同验证，不作为可提交实现。

## 已验证

- `topk_threshold_compact_probe`：CANN CPU 三组形状（7 行单核、15/65 行八核）通过，逐项校验压缩顺序及计数。覆盖全选、全不选、交替、63/64 与 127/128 边界、首尾 64、稀疏掩码。
- `topk_threshold_full_probe`：9 个形状分别使用原生产 probe 的对抗分布、固定种子的正态分布，共 18 个 CANN CPU 用例通过；包含多 batch、尾 batch、多核。
- `topk_threshold_baseline_probe`：相同 18 个 CANN CPU 用例通过。
- 两个完整路径目标的 dav-2201 simulator 设备代码编译通过；原有 cce_global 警告仍存在。没有本轮真实 NPU 或 Judge 结果。

## 测量方法

CANN 9.0.0，Ascend910B1 simulator，一核，完整输入搬运→选择→变换→输出；所有仿真均进行 host 数值比较。随机输入由同一 C++ mt19937/normal_distribution 生成；默认对抗分布含集中最大值、全相同值、末尾大值及混合值，不等于所有行都回退。

耗时采用 profiler 的 duration_time，不将重叠指令耗时累加成 kernel 时间。不能与之前不同输入下的 23.46 us 直接比较，必须使用本轮同输入的 23.08 us 基线。

## 机制观察

随机 T=16 时，基线/候选的指令调用数分别为：VBS32 4/32，VMS4V2 12/64，VREDUCEV2 4/36，VECTOR BAR 78/338。调用次数不是排序数据量：候选单次处理的元素更少，但失去多行统一发射优势，增加采样、阈值同步、压缩和结果暂存。因而减少被排序的元素不能直接转化成同比例 wall time 收益。

## 复现与文件

加载 CANN 环境后，在 WSL/Linux 使用：

```bash
bash tools/run_threshold_probe.sh /path/to/build/threshold cpu
bash tools/run_threshold_probe.sh /path/to/build/threshold sim
```

脚本建立三个 probe 目标并执行正确性及仿真对照。`THRESHOLD_RANDOM=1` 仅控制 host 测试输入，kernel 不读取环境；默认对抗输入。固定阈值 0 的 compact probe 是独立 API 布局检查，不用于估计完整 Top-K 收益。

完整实现：`debug/cpu_topk_probe/threshold_research_kernel.h`；同输入 A/B harness：`debug/cpu_topk_probe/topk_threshold_full_probe.asc`；压缩检查：`debug/cpu_topk_probe/topk_threshold_compact_probe.asc`。研究快照不应在今后 production 改动后自动视为同步更新。

## 正常输入性能结果与决定

| 正态输入行数 | 生产基线 duration / running (us) | 阈值原型 duration / running (us) | duration 改善 |
|---|---|---|---|
| 4 | 6.86 / 6.69 | 6.65 / 6.09 | 3.1% |
| 16 | 23.08 / 22.98 | 21.05 / 20.37 | 8.8% |
| 64 | 87.77 / 87.65 | 77.84 / 76.74 | 11.3% |

六次仿真均通过数值比较。连续 batch 的完整路径收益未达到事先约定的 20%，因此停止本轮 512/256 原型，不接入 submission、不发起新的 Judge 测试，也不继续扫小参数。结论限定于此实现，不等于证明所有阈值算法都不适合该硬件。

## 回退对照及最终结论

16 行默认对抗输入的生产基线 duration/running 为 **23.08/22.98 us**，原型为 **28.96/28.26 us**，duration 增加 **25.5%**，两者输出均通过。这个混合输入包含触发回退的行，不是纯回退路径的隔离测量；但已说明应对这些合法分布时完整原型会明显退化。没有据此宣称所有单行回退都固定慢 25.5%。

[机器可读测量结果](threshold-device-results-2026-10-04.json) 汇总 5 份 CPU 运行日志的通过记录与 8 次仿真数值结果。WSL 实际编译副本与归档源码已做文本一致性核对。

**最终：本轮停止。** 既未达到正常输入至少改善 20% 的门槛，对抗分布的完整路径开销也已超过 10% 的保护目标。不发布新 Judge 候选。此次否定的是 512/256、逐行阈值的具体实现；后续如重新研究 selection，必须先说明如何保留跨行批处理、避免同一串行依赖，而不是继续调本实现的小参数。#48 因总体吞吐问题尚未解决继续开放。
