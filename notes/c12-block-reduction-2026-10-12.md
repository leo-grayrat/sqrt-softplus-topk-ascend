# Case12：基于分块最大值的精确筛选（2026-10-12）

跟踪[#71](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/71)。从PR #70后main开始，用户最近Judge15/15，case12为122.37 us。根据用户TOP20，case12中位数33.13 us，本队慢3.69倍；[差距与研究转向](top20-comparison-2026-10-12.md)。本分支已获完整Judge15/15 Pass，case12为49.24 us（减少59.76%），通过PR #72集成；下文模型与负结果作为实验记录保留。

## 改了选择结构

FP16 E256/k3先精确转为float。每8个输入构成一个分块，通过BlockReduceMax获得32个块最大值。Sort32排序这32个最大值，同时保留不同块的真实ID；Gather取出最大的三个块里的全部24个原值，再以-inf填充到32个位置，第二次Sort32得到最终top3。随后使用原StableSoftplusSqrt和half舍入。

任何未选分块中的值，都不大于三个已保留分块各自的最大值。已保留集合至少包含三个不小于所有被丢弃值的原始元素，因此精确top3分数不会丢失；并列时分数的重复次数保持正确。无需输入分布假设、阈值猜测或候选数读回。Sort32索引在此路径用于分块ID，不能统一写零。

每行Sort32处理256→64个值，去掉原MrgSort合并树；仍读取全部输入，并增加BlockReduceMax与动态Gather，不能据元素数宣称四倍提速。batch上限32，UB约107680 B；Sort32每次至多32 repeat，BlockReduceMax至多128 repeat。除新增类和FP16 experts==256/k==3 gate外，旧生产源码逐字一致。k2/k4研究正确性不表示这些gate已接入；case4仍走原生产类。

## 生产入口完整模型

同源host harness、同输入、同指定核数，CANN9/dav-2201，Ascend910B1 simulator，记录最慢核duration。模型时长不是Judge收益。

|形状|main/us|候选/us|耗时减少|
|---|---:|---:|---:|
|96行/1核|15.24|8.14|46.59%|
|192行/1核|27.76|13.35|51.91%|
|32768行/32核|140.19|58.55|58.24%|
|65行尾batch/1核|11.18|6.81|39.09%|
|1行/1核|3.13|3.79|-21.09%|

96/192行两尺度满足既有20%准入门槛；完整case12形状也进行了输出检查。单行启动成本更高，完整数据保留，不把该路径称为所有规模都更快。

基线全形状第一次5分钟超时，不能使用其未完成数据；一次续接中断同样无效。最后在10分钟时限下重跑完成，原日志保留。重现脚本对32768行使用10分钟；长时限只控制测量进程，不影响kernel。

## 验证层级

- 主机27288组精确分块比较，包括重复值、-inf、极值与最大值集中同一块。
- 研究raw/full/baseline CPU各51/51；raw输出FP32并逐值精确比较，覆盖half最大有限值，不把最终half舍入混入选择检查。
- 接入后双方CPU生产设备dispatch各43/43；CPU因CANN模板注册限制使用同源非模板launch包装，设备sim使用真实run_kernel。
- 最终基线6次、候选12次真实run_kernel仿真输出检查通过，含完整case12、尾batch、多核及E255/k3、E256/k2/k4、E200/k16、E512/k10、E256/k64回归。
- 完整设备编译通过；交付header与实际构建源码逐字一致，双方生产probe一致。旧路径源码核对、结构检查、脚本语法、空白检查通过。

参考容差没有放宽。极大正数的变换舍入#52和负赢家误差#68仍独立开放；raw选择覆盖极值不代表这两个变换问题已修复。上述本地模型验证时尚无Judge；之后用户已回报完整15/15、case12 49.24 us，见同次完整记录。没有本机NPU结果。

## 成本隔离与负结果

|124行/1核诊断|模型/us|
|---|---:|
|main完整路径|18.61|
|仅选择，FP32原始输出|18.30|
|DMA/队列与前k值搬运诊断|3.20|

仅选择与搬运诊断输出合同和FP32写回流量不同，三项不是可加成本拆分；仅搬运不计算TopK，不用于Judge。结果支持选择是主要工作，单独删除变换不足以解决差距。

重复索引最大值归约虽然CPU full/raw各33/33，但96/192行完整模型31.77/47.61 us，main为15.24/27.76 us，因此停止并保留研究源码、脚本和日志。

分块筛选首版逐项填写索引表，96/192行27.20/32.42 us；大量启动标量写入掩盖了计算收益。向量生成表后的研究版本降至7.95/13.17 us；生产入口另行测量，以上表为交付结论，不拼接两个版本的收益。

研究batch48在CPU Gather模型跨页写入处SIGSEGV，batch32通过；没有据此宣称真机存在32KiB Gather限制。Scatter虽有SDK接口，dav_c220实现明确不支持，已撤回。重复归约的偏移源视图在CPU Gather出现读回异常，通过单独根缓冲、srcBaseAddr0和限定视图处理；未禁用检查器。代表性失败日志均归档。

## 复现与交付

先source已安装CANN set_env.sh，再运行：

```bash
bash tools/run_c12_production_probe.sh baseline /absolute/new/baseline-build
bash tools/run_c12_production_probe.sh candidate /absolute/new/candidate-build
```

baseline固定PR #70后main；candidate冻结当前提交。研究、重复归约、成本隔离分别有独立run_c12脚本，诊断生成器固定旧生产基线。冻结目录/home/leo/cann-c12-production-{baseline,candidate}-v2。完整日志、源码校验、指令调用数见[机器可读证据](c12-block-reduction-evidence-2026-10-12/model-summary.json)。指令调用数不是工作量或wall-time分解。

用户完整Judge15/15、误差均0.00%，case12 122.37→49.24 us，减少59.76%。[同次完整15点与阶段集成](current-status-2026-10-12-c12.md)。通过PR #72集成；后续从新main开始case14研究，不在本阶段分支追加生产改动。
