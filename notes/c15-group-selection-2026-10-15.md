# Case15：四路分组最大值精确筛选（2026-10-15）

跟踪[#75](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/75)。从PR #74后的main开始，用户完整Judge15/15，case15 FP32 T65555/E1024/k64仍为1.65 ms；同次平台最优358.05 us。本分支`exp/c15-group-max-selection`是本地验证通过的待Judge候选，尚无本轮真实NPU/Judge数据。已获Judge的C12/C14实现保留。

## 减少参与排序的原值

将一行四个256元素区域的对应位置组成一个四值组，得到256个不同组。三次逐元素Max求各组最大值，Sort32及保留top64的两级合并找到最大值最高的64个组，保留真实组ID。两次Gather取出这64组的全部256个原值，再排序/合并取top64。保持原StableSoftplusSqrt与FP32输出。

对有限实数及-inf，任意未保留值都不大于64个保留组各自的最大值，这64个最大值来自不同原始元素，足以保证精确top64分数的多重集合不变。组内四个值全部保留，赢家集中在同一组或同一区域也安全；并列值不要求额外分布假设。各128列表保留top64再合并同样精确。没有阈值、近似降精度或动态候选数读回。

每行Sort32输入从1024降为256组最大值+256原值=512，合并总量1664→768对，仍搬运全部1024个输入，另增Max和Gather。元素数减少不是速度承诺。batch上限8，UB分配184352 B，含双输入队列、两套pair缓冲、索引与候选表及输出/tmp/pattern；输入根缓冲最大32768 B。两次Sort32分别至多64 repeat，merge至多16 repeat。没有改用仅扩大batch、原合并树不变的旧scratch方案（其模型收益约1.3%，已停止）。

生产只接入FP32 `experts==1024 && k==64`。除新增class与这段gate，整个生产header与PR #74后的main逐字一致；E993..1023的padding路径、k32共享类、generic回退及C12/C14原实现均保留。

## 成本隔离与准入

64行/1核，研究入口同一指定核数和输入：

|诊断|模型/us|
|---|---:|
|旧基线完整|88.19|
|旧基线仅选择，输出原始FP32 top64|87.84|
|旧基线输入搬运及首64列写回诊断|5.36|

仅选择删除数值变换后，耗时仍接近完整路径；仅搬运不计算topk，输出合同不同。这些不是可相加的成本分解，不用原始值/搬运诊断替代Judge提交。

研究完整模型32行45.09→31.71 us（减少29.67%），64行88.19→60.76 us（减少31.10%），满足两尺度20%后再接入生产。随后真实生产入口独立重测如下，不把两个入口的数字混合。

## 真实生产入口仿真

CANN9/dav-2201、Ascend910B1 simulator；双方FP32 CPU和sim均调用实际run_kernel，同源host harness、同输入和指定核数，记录最慢核duration。

|形状|main/us|候选/us|耗时减少|
|---|---:|---:|---:|
|32行 / 1核|45.24|31.65|30.04%|
|64行 / 1核|88.36|60.79|31.20%|
|1043行 / 32核|50.80|39.19|22.85%|
|17行尾batch / 1核|25.23|18.31|27.43%|
|129行尾batch / 8核|25.39|18.55|26.94%|
|1行 / 1核|3.81|4.01|-5.25%|

1043行=1024+19，与完整65555行具有相同的19行全局尾部同余，新batch末批为3行。此形状检查所有输出与32核尾组分配，不能代替完整65555行的吞吐、缓存或Judge测量。上表包含单行和尾batch结果，没有挑选只变快的点；最终比赛收益留待用户完整15点Judge。

## 验证与源审计

- 主机134072组精确分组/prefix比较，含枚举、随机、并列、-inf、有限float极值与16组集中64个赢家。
- 研究full/raw/baseline/baseline_raw/copy CPU各25/25；raw与copy逐值精确比较，覆盖相邻浮点数、极值、跨batch与多核。
- raw设备sim另测17行/1核、129行/8核，逐值精确比较通过，避免最终数值变换容差掩盖选择错误。
- 双方FP32生产CPU各51/51：原46例加入32/64/128/129行以及1043行/32核。包含E993..1024 padding、k32共享类、尾batch、组内/区域集中赢家及很近的FP32分数。
- 基线6次、候选12次真实run_kernel设备sim输出检查通过；候选含E1023/k64、E97/k32与E992/k64、E1024/k63通用回退，以及FP16 E256/k3和E512/k10回归。后两种generic回退沿用原有CPU检查器限制，只在sim验证。
- 候选FP16 CPU另31/31；完整submission设备编译、源范围核对和实测冻结header/生产probe一致性检查通过。FP32完整变换容差沿用2e-5，未放宽checker；#52/#68数值边界仍独立开放。

研究probe最初沿用旧large-k probe的宽泛host CLI形状校验，但所有实测均明确输入E1024/k64；交付时收紧为exact形状，在新目录重建raw后25/25通过，并确认E1000/k64被拒绝。性能版本与交付研究probe仅此host guard不同，device class未变；差异和单独验证已记录于证据JSON，不把未重测的host源码称为逐字一致。正式生产probe与交付版本一致，支持所有其回归形状。

[完整日志、源码校验与core0指令调用数](c15-group-selection-evidence-2026-10-15/model-summary.json)。指令调用数不是元素量或wall-time分解。所有数据来自完整结束的进程，失败未被隐藏；本轮没有通过放宽数值比较来获得通过。

## 复现与交付

先source本机安装的CANN set_env.sh，在仓库使用全新的绝对构建目录：

```bash
bash tools/run_c15_research_probe.sh /absolute/new/research-build
bash tools/run_c15_production_probe.sh baseline /absolute/new/baseline-build
bash tools/run_c15_production_probe.sh candidate /absolute/new/candidate-build
python3 tools/check_c15_group_selection.py
python3 tools/check_c15_source_scope.py
```

研究及生产baseline固定PR #74后的main，诊断clone保留该版本选择代码；candidate冻结当前submission。目录`/home/leo/cann-c15-group-v1`、`/home/leo/cann-c15-production-{baseline,candidate}-v1`、`/home/leo/cann-c15-guard-v1`保留原始构建与结果。

用户只需替换本分支`submission/dense_repeat_override.h`并跑完整15点Judge。PR保持Draft、#75保持打开，完整正确性及收益确认后再合main；不在main中覆盖已验证基线，不拼接不同分支单点最优。
