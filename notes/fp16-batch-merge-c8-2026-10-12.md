# FP16 case8：合并批量化、原生half排序均停止（#67）

以PR #65已Judge15/15的main为基线，case8 T4096/E256/k64最新33.77 us。固定batch16、UB分配、真实候选集合、FP32变换、half舍入。两个候选均未达到连续batch两个尺度20%门槛，生产已完全撤回此阶段。

## 两个候选

1. 合并候选：两条有序length64列表拆四条length32，跨行最终完整合并；两列表prefix平面化为一次拷贝。Sort32和中间合并的元素数量不变。
2. 在合并候选上，E256连续行使用原生half Sort32/MrgSort，取消全部输入Cast。pair记录仍占8字节，half值占其中一项，32位index为0payload；地址步长和prefix字节数不减。GatherMask使用每4个half取一个值的128bit pattern，先收集half到独立tmp，再Cast选中的64个值为float，沿用原FP32变换。填充行仍走float选择路径。

两条生产候选均限制在HalfLarge已有gate（inner256/k64），没有修改FP32或HalfMedium。相同安装版API已在main原生half小k路径使用；本次通过完整生产路径另行验证。

## 完整模型负结果

CANN9.0/dav-2201、Ascend910B1 simulator，真实run_kernel包含host tiling、GM读写、选择与变换，单次kernel duration，不含H2D/host时间。

| 形状 | main/us | 合并候选/us | 原生half候选/us |
|---|---:|---:|---:|
| 64行/E256/k64/1核 | 23.67 | 23.62 | 23.66 |
| 128行/E256/k64/1核 | 44.89 | 44.64 | 44.54 |
| 1行/E256/k64/1核 | 3.13 | 3.31 | 3.39 |
| 4096行/E256/k64/32核，max core | 44.92 | 44.67 | 44.59 |
| 257行/E241/k64/8核，max core | 14.19 | 14.22 | 14.22 |
| gate外257行/E128/k16/8核，max core | 5.51 | 5.73 | 5.72 |

64/128行收益分别仅0.21%/0.56%（合并）和0.04%/0.78%（原生half）；单行均变慢。无真实NPU或新Judge收益。

| 64行指令调用数 | main | 合并 | 原生half |
|---|---:|---:|---:|
| VBS32 | 4 | 4 | 4 |
| VMS4V2 | 68 | 8 | 8 |
| VCONV | 8 | 8 | 8 |
| MOV_UB_TO_UB | 12 | 8 | 8 |
| BAR | 152 | 76 | 88 |

VCONV调用相同但原生half从全量输入转换换成选中值转换；调用数不代表元素量。MrgSort调用虽大减，其列表元素工作量未减，模型未体现整体提速。后续应减少真正的合并工作量，避免继续只削发射数。

## 验证和保留物

基线、合并、原生half的CPU生产dispatch各45/45。CPU调用生产dispatch包装，simulator直接调用真实run_kernel；CANN9 CPU half模板模式注册限制沿用main记录。基线6次、两候选各9次sim输出检查全通过，两候选完整设备编译通过。实际输入包含重复值、集中赢家、尾列赢家、-inf、±10000、填充E225/239/240/241/255/256、batch16边界与多核尾部。参考函数和0.5 half ULP + 2e-6容差均未放宽。

最初集中fixture30-i使top64包含-12，两版均严格数值失败；[原始失败](fp16-batch-merge-c8-evidence-2026-10-12/negative-input-failure/README.md)由#68独立跟踪。之后集中fixture100-i，不宣称该负赢家的变换已修复。#52极大正数舍入问题也未纳入本阶段。

仅保留[合并研究补丁](../debug/fp16_c8_batch_merge_only.patch)、[原生half研究补丁](../debug/fp16_c8_native_sort.patch)、[固定main复现脚本](../tools/run_fp16_batch_merge_c8_probe.sh)、[生产probe](../debug/cpu_topk_probe/topk_fp16_large_production_probe.asc)、[原生half证据](fp16-batch-merge-c8-evidence-2026-10-12/model-summary.json)和[合并证据](fp16-batch-merge-c8-evidence-2026-10-12/merge-only/model-summary.json)。脚本baseline/candidate/merge-only均从固定main生成，candidate施加原生half补丁。原始跑测脚本冻结于各WSL构建目录，不与后来生产源码混用。#67保持开放为研究停止记录；无需新Judge。
