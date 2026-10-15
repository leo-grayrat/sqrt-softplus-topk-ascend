# FP32 k8：分块最大值精确筛选（2026-10-15）

跟踪[#77](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/77)。从PR #76后的已验证main开始，活动分支`exp/fp32-k8-block-selection`。该main完整Judge15/15，case3 T7777/E888/k8为115.17 us、case13 T32999/E210/k8为132.28 us。当前候选只接入E888/k8，本地验证完成，等待用户完整15点CANNJudge；尚未合并main。

## 选择结构

每行按连续8值分块，BlockReduceMax取各块最大值，按这些最大值精确保留8个不同块，再Gather这8块的全部64个原始FP32值，Sort32/合并取top8。原StableSoftplusSqrt、FP32输出不变，没有数据分布假设、阈值、动态候选上限或近似降精度。

任意未保留块的任意值都不大于8个保留块的最大值。它们来自8个不同原始元素，因此未保留值不能改变top8分数多重集合。8个赢家集中同一块时，该块全部8值仍被保留；并列值同样成立。本轮证明和测试覆盖有限FP32及-inf，不扩大既有数值合同。

源行显式初始化到二次幂长度，E888补至1024、E210补至256，所有填充为-inf，块ID保留真实来源。即便只1个有限值或整行-inf，选到填充块也只读取已初始化的合法缓冲，不把有限值重复构造成多个候选。

|形状|旧Sort32输入/行|新Sort32输入/行|旧merge输入对/行|新merge输入对/行|batch上限|UB分配/B|
|---|---:|---:|---:|---:|---:|---:|
|E888/k8|896|128+64=192|304|192|8|131104|
|E210/k8|224|32+64=96|80|64|32|180256|

E210行仅描述研究实现，未获准入则不用于生产。旧merge计入补齐槽位和最终split-list合并；这些是输入元素量，不是指令调用数或耗时分解。batch上限与旧复用scratch路径相同，收益来自减少排序/合并工作；仍搬运全部输入，另增块归约、索引和Gather。输入根源张量上限32768 B，遵循前期CPU Gather模型的可复现配置；不声称这是硬件限制。

## 研究准入

同源host、指定单核，完整选择和数值变换都保留，每个E两个尺度至少减少20%才允许接入生产：

|行数|E|main研究入口/us|候选研究入口/us|耗时减少|
|---:|---:|---:|---:|---:|
|32|888|20.03|11.92|40.49%|
|64|888|37.85|20.54|45.73%|
|64|210|12.15|12.36|-1.73%|
|128|210|21.61|20.06|7.17%|

E210未满足两个尺度均减少20%，仅归档研究负结果，生产保留旧路径。准入配置归档于[JSON](../debug/fp32_k8_block_admission.json)，代码范围检查同时验证阈值及实际gate。研究baseline直接实例化实际小k路径的SqrtSoftplusVendorPerfTopKKernel，候选实例化新类。下面另以真实run_kernel入口独立重测，不混用两种入口的计时。

## 真实生产入口仿真

CANN9/dav-2201，Ascend910B1 simulator，记录最慢核duration。双方同一host输入、完整逐行参考比较、相同指定核数。包含完整case3和case13、32核的全部输出检查：

|形状：r行/e列/k/c核|main/us|候选/us|耗时减少|
|---|---:|---:|---:|
|r1-e210-k8-c1|4.08|4.22|-3.43%|
|r1-e888-k8-c1|3.83|4.59|-19.84%|
|r128-e210-k8-c1|22.01|22.29|-1.27%|
|r32-e888-k8-c1|20.22|12.97|35.86%|
|r32999-e210-k8-c32|160.42|160.82|-0.25%|
|r64-e210-k8-c1|12.55|12.83|-2.23%|
|r64-e888-k8-c1|38.03|21.60|43.20%|
|r7777-e888-k8-c32|141.14|71.85|49.09%|

单行回退和所有多尺度结果原样保留。E210生产走旧路径，其整形状模型变化也照实列出，不能因源码范围未变而省略回退数据。模型绝对时间和比例不等于真实NPU的CANNJudge，当前不据此承诺提交后的性能。

## 验证与失败记录

- 主机141078次精确分块/padding比较：小域枚举、随机、并列、有限float极值、0/1/7个有限值、8个赢家同块。
- 研究raw/full/baseline CPU各38/38。raw逐值精确比较，完整变换容差沿用2e-5，没有放宽。含1..129行的跨batch、17/129/257行多核、相邻FP32、尾部赢家、全-inf。
- 研究完整full/base共8次sim，raw另17行/E888/1核与129行/E210/8核sim逐值精确比较，均通过。
- 双方实际run_kernel FP32 CPU各49/49，含相邻E、k7、medium-k、C15和32核尾组；基线8次、候选14次FP32设备sim通过，包括T7777/E888与T32999/E210完整形状各32核。
- 候选另FP16 CPU31/31，C12 E256/k3与C14 E512/k10设备sim通过；FP32回归含E1024/k64 C15，以及k9通用回退。通用回退沿用既有CPU检查器限制，只在sim检查。
- 完整submission设备编译通过，交付header与LF交付快照逐字一致；早期性能测量构建的header/probe与交付版在LF归一化后逐字一致，证据同时记录原始和归一化源指纹（Windows CRLF经Git规范为LF）。另从暂存区导出全新LF快照，CPU49/49、129行/E888/k8/8核sim及完整设备编译通过，不用此补测替代前期性能数据。删除新类和exact gate后，整个header与PR #76后的main逐字一致。kernel.asc、reference和既有C12/C14/C15代码没有改动。

研究v1基线曾错误调用另一条TagFloat dispatch并传空tiling，CPU报零长InitBuffer并失败。实际FP32小k由host提前分流至独立prefix dispatch。v1不计入任何基线计时，失败日志和原wrapper保留于[evidence/failures](fp32-k8-block-evidence-2026-10-15/failures)。修正研究入口后在全新v2目录重建，三个CPU套件与全部模型重新完成；实际生产probe始终调用run_kernel，其基线49例独立通过。#52/#68数值边界继续独立开放。

[全部计时、逐核数量、源指纹与core0指令调用数](fp32-k8-block-evidence-2026-10-15/model-summary.json)。指令调用数不能直接当作耗时分解。原始完整build/run目录保留在WSL的`/home/leo/cann-fp32-k8-block-v2`和`/home/leo/cann-fp32-k8-production-{baseline,candidate}-v1`。

## 复现和待测文件

先source安装的CANN set_env.sh，再使用新的绝对构建目录：

```bash
bash tools/run_fp32_k8_block_research.sh /absolute/new/research-build
bash tools/run_fp32_k8_block_production.sh baseline /absolute/new/baseline-build
bash tools/run_fp32_k8_block_production.sh candidate /absolute/new/candidate-build
python3 tools/check_fp32_k8_block_selection.py
python3 tools/check_fp32_k8_block_scope.py
```

研究与生产baseline固定PR #76后的main，candidate冻结当前submission。准入JSON是已测冻结版本的记录，不会自动使用新目录更新；复跑后应比较实测计时再审计gate。

用户只需替换[submission/dense_repeat_override.h](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/blob/exp/fp32-k8-block-selection/submission/dense_repeat_override.h)，运行完整15点Judge并回传结果。获得完整Judge后再收尾PR合入main；当前main仍是已验证的C15版本。
