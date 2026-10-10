# FP32 k=8 scratch 复用：实现与完整路径测量（#55）

本轮快速落实用户研究报告的第一优先项，只改缓冲生命周期和 batch 上限，没有同时改 merge tree 或数值变换。结论：正确性与编译通过，E=210 完整模型改善 2.3–3.1%，E=888 约 10%，均未达到预设 20% 门槛，因此保留研究补丁，不接入生产、不要求新 Judge。

## 实际实现

基线为报告归档后的 main。修改 `SqrtSoftplusVendorPerfTopKKernel`，限定 FP32、k=8、inner=224/896：

- pairB 前半部用于输入 padding/rearrange，后半部用于零索引；Sort32 的 PIPE_V 屏障之后，整块恢复为 merge scratch。
- 这条路径不再分配独立 inputBuffer/indexBuffer。由于 merge 会复用 pairB，每批重新初始化索引，不能沿用原来仅初始化一次的方式。
- inner=224 的最大 batch 从 16 到 32；inner=896 从 4 到 8。两者最多 224 Sort32 repeats。短输入仍按每核行数缩小 batch。
- 其他 dtype/k/inner 不改变算法和 batch 策略。输入预取、合并树、变换和写回沿用生产代码。

最大 batch 的显式 UB 分配（含双缓冲、pair、输出和临时区；不含框架额外开销）：

| 形状 | main | 不复用而直接翻倍 | 复用并翻倍 |
|---|---:|---:|---:|
| E=210/k=8 | 114720 B | 229408 B | 172064 B |
| E=888/k=8 | 114720 B | 229408 B | 172064 B |

复用确实让更大 batch 的显式存储低于 192 KiB，但并不删除排序或合并的总数据工作。

## 验证与结果

同一个 probe 直接调用 `run_kernel`，通过真实 host 分流进入 FP32 small-k 生产路径；没有使用复制算法或不适用的 float 高阶 TopK dispatch。baseline/candidate 仅 submission 内容不同。

- CPU：双方各 **38/38**。覆盖 E210 的 1/15/16/17/31/32/33/64/128 行、E888 的 1/3/4/5/7/8/9/16/32 行，目标 inner 的边界 E、邻近未修改 gate、k=7 回退、多核和尾批次。输入包含确定性混合数据、重复值、全 -inf、单段极值、尾部极值、跨 32 元素块并列值以及 ±10000。
- 这里验证的是完整变换结果，采用已有 FP32 probe 的误差要求；没有另做 raw Top-K 的独立逐位输出检查。
- 双方 sim 设备编译通过；候选完整 submission 的 dav-2201 编译通过。
- Ascend910B1 simulator：同输入、单核、k=8，12 次模型运行均通过完整输出比较。下表为每个配置单次模型的 duration，不是真实 NPU/Judge，也不是统计分布。

| E | 行数 | main μs | 候选 μs | 耗时减少 |
|---:|---:|---:|---:|---:|
| 210 | 64 | 16.56 | 16.18 | 2.29% |
| 210 | 128 | 30.46 | 29.51 | 3.12% |
| 888 | 16 | 13.55 | 12.20 | 9.96% |
| 888 | 32 | 24.62 | 22.19 | 9.87% |
| 210 | 1 | 3.77 | 3.75 | 0.53% |
| 888 | 1 | 5.16 | 5.09 | 1.36% |

## 指令证据与解释

| E / 行数 | Sort32 调用 main→候选 | MrgSort 调用 main→候选 |
|---|---:|---:|
| 210 / 64 | 4→2 | 8→4 |
| 210 / 128 | 8→4 | 16→8 |
| 888 / 16 | 4→2 | 32→24 |
| 888 / 32 | 8→4 | 64→48 |

计数取自模型指令表 VBS32/VMS4V2 的 call_count 汇总，调用次数不等于总 repeats 或总工作量。batch 扩大确实生效，但每行排序和候选处理量未减少；E888 的最终两路合并仍逐行执行，其调用不能随 batch 翻倍而减半。因此此次数据只支持“减少发射有小幅收益”，不支持把 buffer 复用单独作为大幅性能突破。

结构文本 checker 曾因 `input = inputBuffer_...` 改成条件选择别名而失败，是旧精确字符串规则不能识别新写法；真实生产 CPU 输出检查通过。归档并恢复生产代码后原 checker 恢复通过，没有放宽检查来宣称候选通过该文本规则。

## 复现和阶段决策

- [可应用研究补丁](../debug/fp32_k8_scratch_reuse.patch)：包含实际测量的全部生产修改；仓库 submission 恢复为 main 的已验证版本。
- [生产入口 probe](../debug/cpu_topk_probe/topk_fp32_scratch_production_probe.asc)。
- [复现脚本](../tools/run_fp32_scratch_probe.sh)：加载 CANN set_env.sh 后，分别执行 `bash tools/run_fp32_scratch_probe.sh baseline /absolute/baseline-build` 和 `bash tools/run_fp32_scratch_probe.sh candidate /absolute/candidate-build`。脚本从固定基线导出 submission，候选再应用补丁。
- [日志与指令计数](fp32-scratch-evidence-2026-10-10)：归档日志只去掉行尾空格。

本轮仅检验报告第一优先项，报告第二项“按真实叶子数和布局省掉中间合并/搬运”尚未实现；#52 仍独立。缓冲复用可供未来结构改造复用，但不单独提升为 Judge 候选。研究记录通过阶段 PR 归档回 main，#55 保留开放记录；生产仍为此前 15/15 Judge 基线。
