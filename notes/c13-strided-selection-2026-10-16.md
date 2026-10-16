# Case13：四值跨段分组与紧凑top8合并（2026-10-16）

跟踪[#79](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/79)。从PR #78后的已验证main起步，该main完整15/15，case3为53.56 us、case13 T32999/E210/k8为132.01 us。本阶段分支`exp/c13-strided-group-selection`仅接入FP32 E210/k8，本地验证完成，等待完整CANNJudge；main仍是已验证的PR #78版本。

## 改变分组和中间列表大小

上一轮连续八值块方案对E210完整研究模型只有-1.73%/+7.17%，因此未接入生产。此次将E210初始化补齐至256，四个64列区域的对应位置组成64个四值组；3次重复Max使用BinaryRepeatParams跨行批量执行，不逐行发射。每组最大值连真实组ID一起排序；两份sorted32各只留top8，再将16对视为四个sorted4批量合并。从前8个不同组Gather全部32个原始值，Sort32后取top8。

若某组被舍弃，其所有值都不大于8个保留组的最大值；这8个最大值来自不同原始元素，故保留组的全部成员足以保持精确top8分数多重集合。并列、同组集中赢家及仅一个有限值同样成立。所有填充位置显式为-inf，所有组ID指向合法初始化源，原StableSoftplusSqrt与FP32输出保留。合同仍为有限FP32及-inf，不借此扩大既有数值边界。

|结构|main|四值分组候选|
|---|---:|---:|
|Sort32输入值/行|224|64组最大值+32原值=96|
|MrgSort输入对/行|80|16|
|Gather原值/行|0|32|
|batch上限|32|32|
|UB分配/B|172064|163872|

仍读取全部210个输入，增加分组、索引和Gather。工作量数字不直接等于耗时；收益不来自扩大batch。选择源根缓冲上限32KiB，保持前期CPU Gather可复现配置，不将该模型现象称为硬件限制。

## 研究与准入

每个方案均运行完整选择、数值变换和输出比较，两个单核尺度都至少减少20%才准入。

|方案|行数|main研究入口/us|候选/us|耗时减少|
|---|---:|---:|---:|---:|
|四值跨段组|64|11.98|9.27|22.62%|
|四值跨段组|128|21.44|14.34|33.12%|
|八值块+紧凑合并|64|11.98|10.52|12.19%|
|八值块+紧凑合并|128|21.44|16.58|22.67%|

仅四值分组达到双尺度门槛。八值块诊断保留了上一轮分组，只把最终64对合并改为各sorted32留8、合并16对，未满足两个尺度均20%，不接入生产。它与四值方案都排序96个值、合并16对，但分组、归约和Gather数量不同；这是完整路径对比，不作可相加的成本分解。[准入记录](../debug/c13_selection_admission.json)。

## 实际生产入口

双方CPU/sim都调用真实run_kernel，指定相同输入、核数、CANN9/dav-2201与Ascend910B1 simulator；记录最慢核duration，检查全部输出。

|形状：r行/e列/k/c核|main/us|候选/us|耗时减少|
|---|---:|---:|---:|
|r1-e210-k8-c1|4.22|3.86|8.53%|
|r128-e210-k8-c1|22.29|14.92|33.06%|
|r129-e210-k8-c8|6.09|5.87|3.61%|
|r257-e888-k8-c8|13.97|13.91|0.43%|
|r32999-e210-k8-c32|160.82|89.49|44.35%|
|r64-e210-k8-c1|12.83|9.85|23.23%|
|r65-e1024-k64-c8|11.20|11.55|-3.13%|

完整T32999/E210/k8、32核模型为160.82→89.49 us，减少44.35%。单行、尾batch及已有C3/C15回归原样列出，不只展示加速点。模型比例和绝对时长均不能替代用户真机Judge；当前不预测Judge时间。

## 验证与范围

- 主机141072次精确比较，含小域枚举、随机、有限float极值、-inf/稀疏有限值及prefix裁剪。
- 四值full/raw、八值block/block_raw、baseline研究CPU各19/19，覆盖1..129行、跨batch、17/129/257行多核、相邻FP32、全-inf、同一连续块及跨段组集中赢家；raw逐值精确比较，完整变换仍使用原2e-5容差。
- 三种完整研究路径的64/128行共6次sim，以及两种raw的17行/1核、129行/8核共4次设备sim，全部输出比较通过。
- 双方实际生产FP32 CPU各49/49；基线7次、候选11次FP32 sim均通过，包括完整case13全部32999行/32核、尾batch、相邻E/k、k9 generic，以及C3 E888/k8、C15 E1024/k64回归。generic路径沿用CPU检查器限制，只用sim检查。
- 候选FP16 CPU31/31，C12 E256/k3及C14 E512/k10两次sim通过。全阶段30次设备sim输出检查，完整submission设备编译通过。
- 新类与研究设备代码一致（仅重命名、移除raw宏），exact E210/k8 gate放在真实FP32低k prefix dispatch。移除新类及gate后，整个生产header与PR #78后main逐字一致，已有C3/C12/C14/C15均保留。生产只改dense_repeat_override.h，没有改变kernel.asc或官方reference。

研究冻结之后才追加生产CMake target，研究构建的CMake与交付版差异仅为未使用的生产target；实际研究类、probe及编译选项未变。所有原始与LF归一化源指纹、完整CPU/编译/模型日志及core0指令调用数均归档于[证据](c13-strided-selection-evidence-2026-10-16/model-summary.json)。指令调用数不是时间分解。#52/#68独立数值问题继续开放。

## 复现与待测

先source安装的CANN set_env.sh，使用全新绝对构建目录：

```bash
bash tools/run_c13_selection_research.sh /absolute/new/research-build
bash tools/run_c13_selection_production.sh baseline /absolute/new/baseline-build
bash tools/run_c13_selection_production.sh candidate /absolute/new/candidate-build
python3 tools/check_c13_strided_selection.py
python3 tools/check_c13_selection_scope.py
```

研究和生产baseline固定PR #78后的main，candidate冻结当前submission。准入JSON为本轮冻结结果，复跑时需核对新计时再判断。原目录为WSL的`/home/leo/cann-c13-selection-v1`、`/home/leo/cann-c13-production-{baseline,candidate}-v1`，失败或负结果不被成功运行覆盖。

只需替换[submission/dense_repeat_override.h](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/blob/exp/c13-strided-group-selection/submission/dense_repeat_override.h)，运行完整15点CANNJudge。完整Judge确认后再收尾PR合入main；本阶段issue在此之前保持开放。
