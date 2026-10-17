# C15逐行排序阈值：待完整Judge

跟踪[#85](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/85)，分支`exp/c15-order-statistic-filter`，仅替换`submission/dense_repeat_override.h`。main仍保留用户确认的1.02ms基线；上一矩估计版15/15但1.26ms、慢23.53%，PR #86禁止合入。完整旧回报见[c15-batch-filter-judge-2026-10-17.json](c15-batch-filter-judge-2026-10-17.json)。

本轮取消mean+1.35*std预测。将每行1024值分成128个互不相交的8值跨段组，取第64大组最大值作为该行阈值。64个不同组各提供至少一个达标原值，故至少64个原值达标，不需要正态或均匀假设。仍逐批精确统计，任一行超过128候选或阈值非有限则执行原路径；并列/集中分组输入仍可能回退，性能不保证与分布无关。

每行阈值通过128值Sort32及一次128对合并得到，后续沿用批量计数/整批压紧/128槽恢复/原值排序。成功路径总Sort32输入256值、合并256对，基线为512值/768对。保留FP32原值、原变换、输出、batch10及双输入队列；Normal掩码打包索引每core初始化一次，新增5120字节缓存，总UB181792字节（低于192KiB），其余缓冲不变；C15类外与固定main一致。

## 实际生产入口同条件A/B

CANN9/dav-2201/Ascend910B1 simulator，多核取最慢核；“减少”为正，负数表示变慢。基线固定PR #84后的main。模型不代表真实NPU或Judge收益。

| 输入及规模 | main us | 候选us | 耗时减少 |
| --- | ---: | ---: | ---: |
| 均匀128行/1核 | 105.06 | 76.34 | +27.34% |
| 均匀1043行/32核 | 34.86 | 26.76 | +23.24% |

## Normal修正前研究实现A/B（诊断记录）

| 输入及规模 | main us | 候选us | 耗时减少 |
| --- | ---: | ---: | ---: |
| 正态64行/1核 | 53.62 | 37.80 | +29.50% |
| 均匀64行/1核 | 53.68 | 37.80 | +29.58% |
| 对抗64行/1核 | 58.63 | 69.59 | -18.69% |
| 逐行偏移64行/1核 | 53.62 | 37.80 | +29.50% |
| 均匀1043行/32核 | 34.55 | 24.56 | +28.91% |

4次最终生产成对模型和10次Normal修正前研究模型输出均通过；上表研究数据不冒充最终源码的时间。对抗回退增加阈值排序开销，具体负数原样保留，不能只报告快路径。前一个Normal动态索引版（production-v3）66组通过，32核34.85→28.06us、仅减少19.48%，未达20%准入门槛。其256行候选完整输出通过、161.27us，但main基线超时，只给截断时长，不形成A/B；128行/32核补测使用原二进制，完成日志与原超时日志分开。最终production-v4将与输入无关的mask索引移到每core初始化，并重跑上述生产验证，不把上一版CPU或模型冒充最终版数据。设备完整submission编译通过。Normal修正前研究CPU full/raw各225组、共450组通过，但不含23行；实际生产随后在23行均匀输入的GatherMask CPU SetInput/PV_MEM写入处SIGSEGV，独立重复两次仍失败，而8/9/10行通过，原失败保留。尝试rows==batch_保护仍失败，v2也未补COMPLETE。最终改公开Normal模式、64值repeat，紧凑mask每64bit扩成32字节槽，布局由独立1100行mask检查确认，没有禁用CPU检查器或添加CPU专用算法。CPU库Counter输入拷贝长度过大的反汇编线索保留为待运行时确认的解释，不当成已证实根因或硬件限制。最终生产原失败23行1组、默认51组、额外8种分布和1043行正态1组，加raw 13/23/33行均匀、23行偏移/超限共5组，合计66组通过。raw直接验证原值选择，参考与容差未变。

主机8000行检查中，旧矩估计在均匀100批仅54批命中、逐行偏移0批，新方案在两者均100/100；并列、集中分组、非有限输入的回退保留。旧版实际生产均匀64行模型54.10→58.39us也退化7.93%，支持分布敏感机制，不能据此断定Judge退化全部由此造成。官方starter仅case0使用uniform(-5,5)，未掌握隐藏Judge输入，不宣称其分布已知。

独立只读审查无代码阻塞项；范围checker确认C15类外与main一致，研究/raw去诊断后与生产一致。原始日志、冻结变更源码、各核时间、manifest见[c15-order-filter-evidence-2026-10-17/summary.json](c15-order-filter-evidence-2026-10-17/summary.json)。公共工程由摘要固定commit复原，未重复复制全仓。首个研究驱动在发现baseline将引用候选header后主动停止，未使用其A/B；v2改为git archive固定main后全新运行，停止说明保留。

复现前source本机CANN9环境，输出目录须不存在；编译-j1：

```bash
bash tools/run_c15_order_filter_research.sh /home/leo/new-order-research
bash tools/run_c15_order_filter_production.sh /home/leo/new-order-production
python tools/check_c15_order_filter.py
python tools/check_c15_batch_filter_scope.py
```

派生脚本已语法检查，实际运行的驱动快照一并归档。完整65555行和15点硬件回归仍需用户CANNJudge，issue保持开放，候选不合main。
