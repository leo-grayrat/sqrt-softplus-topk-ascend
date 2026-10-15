# Case14：精确分块最大值筛选（2026-10-12）

跟踪[#73](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/73)。从PR #72后main开始，用户同一次完整Judge15/15、case12为49.24 us，case14 FP16 T65536/E512/k10为536.41 us。本分支`exp/c14-block-max-selection`已于2026-10-15获用户完整Judge15/15、误差均0.00%，case14为326.76 us（减少39.08%），通过PR #74集成。[同次完整结果](current-status-2026-10-15-c14.md)。下文为前期本地模型记录，本机无NPU。

## 选择结构与精确性

将512个half精确转为float，每8个值归约一个最大值，得到64个分块最大值。Sort32两组并等长四路合并这些最大值，保留其真实分块ID。取最大的16个不同块，Gather其全部128个原值，按四组Sort32排序，每组保留16个候选，再一次等长四路合并取top10，使用原StableSoftplusSqrt与half舍入。

任意未保留的值，都不大于16个保留块各自的最大值。这16个最大值来自不同原始元素，数量至少为k10，故被丢弃值不能改变精确top10分数的多重集合；并列值的重复次数同样保留。每个候选32值叶子保留16>=10也安全。此结论不依赖输入分布、阈值或采样，不读回动态候选数。

Sort32每行处理512→192个值，MrgSort总合并240→128对；增加BlockReduceMax和两次动态Gather，仍读取全部输入，不能按元素数推出实际速度。batch上限16，UB分配166816 B（k10，含双输入队列）；输入float根缓冲32768 B。BlockReduceMax至多128 repeat，最大值/候选Sort32至多32/64 repeat，MrgSort至多16 repeat。索引表在启动时按向量生成，不对每个元素逐项写入。

生产gate仅half `experts==512 && k==10`，其余完整生产源码与PR #72后的main逐字一致，包括已获Judge的case12路径。研究k9/k12通过不表示它们接入生产。

## 真实生产入口仿真

CANN9/dav-2201、Ascend910B1 simulator，同源host harness、同输入、同核数；双方sim均调用真实`run_kernel`。记录每次最慢核duration，而非Windows计时或CPU模拟耗时。

|形状|main/us|候选/us|耗时减少|
|---|---:|---:|---:|
|64行 / 1核|25.01|19.55|21.83%|
|128行 / 1核|46.91|35.64|24.02%|
|4096行 / 32核|49.35|35.74|27.58%|
|33行尾batch / 1核|14.32|12.09|15.57%|
|249行尾batch / 8核|16.30|11.53|29.26%|
|1行 / 1核|3.31|4.12|-24.47%|

64/128行分别减少21.83%/24.02%，满足两尺度20%准入。4096行包含每核8个完整batch，32核最大耗时减少27.58%，并检查所有行输出。单行增加0.81 us，原样保留，不能称所有规模更快。

未运行完整65536行仿真：本轮32核仿真覆盖4096行，而Judge形状每核有128个batch。大规模设备时间、流水稳态和最终比赛收益仍由用户完整Judge验证，不把27.58%外推为Judge承诺。

## 验证与证据

- 主机27288次精确分块/prefix比较，包括并列、-inf、half有限极值与赢家集中同一叶子。
- 研究raw/full/baseline CPU各51/51；raw输出FP32、逐值精确比较，k9/10/12、batch前后、尾batch、多核、极值与很小输入均覆盖。完整变换用原严格half容差。
- 接入后双方CPU设备dispatch各31/31，CPU沿用非模板包装；sim走真实run_kernel，不能混称CPU测试验证了host launch。
- 基线6次、候选12次设备sim均通过输出检查，含单行、两尺度、尾batch、多核、4096行/32核、E511/k10、E512/k9/k12、E256/k3（已验证case12）、E200/k16、E256/k64回归。
- 候选完整submission设备构建通过；交付header与实测冻结源码相同，双方生产probe相同；源范围核对通过。没有改reference或放宽checker，#52/#68的共有数值边界仍独立开放。

研究入口模型64行24.79→19.46 us、128行46.82→35.55 us，随后生产入口重新独立测得上表，不混合使用不同入口的数字。

完整日志、源码校验及core0指令调用数见[机器可读证据](c14-block-reduction-evidence-2026-10-12/model-summary.json)。指令调用数不等同于工作量或时间拆分。

## 失败与修正

首版选择12块、96候选，最后合并用了16/16/8/8的四路长度并跨行repeat。第1行通过，第2行raw结果-20而期望-4；停止该布局，改用16块128值、四路等长16，CPU51/51与上述所有设备检查通过。原始失败源码和日志保留于证据failures目录；不把单行正确性当成跨行正确性。

首个生产probe目标遗漏架构编译选项和链接配置，基线/候选均报`LLVM ERROR: PlugIn Error: no aicore arch specified.`。补齐既有probe同样的`--npu-arch`与库设置，在新冻结目录重建成功；未把这个构建配置错误归因于算法或更换生产API。失败日志保留。

## 复现与交付

先source安装的CANN set_env.sh，再在仓库运行；每次使用全新绝对构建目录：

```bash
bash tools/run_c14_research_probe.sh /absolute/new/research-build
bash tools/run_c14_production_probe.sh baseline /absolute/new/baseline-build
bash tools/run_c14_production_probe.sh candidate /absolute/new/candidate-build
python3 tools/check_c14_block_selection.py
python3 tools/check_c14_source_scope.py
```

研究和生产baseline均固定PR #72后的main，candidate冻结当前submission；研究baseline不会意外调用当前候选gate。冻结目录`/home/leo/cann-c14-block-v2`、`/home/leo/cann-c14-production-{baseline,candidate}-v2`。启动后台进程后发生用量中断，继续时确认双方COMPLETE，归档只使用实际完成的日志。

2026-10-15用户已回报完整15/15 Judge，case14 536.41→326.76 us，减少39.08%。通过PR #74集成本阶段，#73在用户验证后关闭。收尾不再改用户已测header；[同次完整结果与阶段记录](current-status-2026-10-15-c14.md)。C12结果单独归档，不将不同候选单点最优拼接为本轮结果。
