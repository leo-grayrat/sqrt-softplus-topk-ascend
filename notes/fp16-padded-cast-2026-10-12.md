# FP16 填充行跨行Cast：停止生产实验（#66）

基线为PR #65已验证main；用户case5最新6.58 us。仅对HalfMedium inner224/k9..16的非连续填充输入，以4个64列条带跨行Cast替代逐行Cast。输入重复步长13个32B block，输出28个block，末条带16元素。排序、候选、batch、UB与舍入不变。

CANN9/dav-2201公开Cast mask/repeat接口和UnaryRepeatParams已从安装头文件核对；实际设备实现vconv_f162f32接受独立输入/输出重复步长。CPU生产dispatch双方76/76，真实run_kernel simulator双方全部16个输出检查通过（基线6、候选10），候选完整设备编译通过。

| 形状 E200/k16 | main/us | 候选/us | 耗时减少 |
|---|---:|---:|---:|
| 64行/1核 | 15.75 | 15.25 | 3.17% |
| 128行/1核 | 28.97 | 27.73 | 4.28% |
| 1行/1核 | 2.94 | 3.21 | -9.18% |
| 512行/32核，最大core | 6.06 | 6.02 | 0.66% |

均为单次完整kernel模型，无真实NPU或新Judge测量。64行VCONV调用66→10、BAR114→58，但完整耗时只减少3.17%；调用数减少不能当作速度倍数。非目标路径同源回归E128/k16 257行/8核5.51→5.84 us，E512/k10 121行/8核7.89→8.24 us；代码布局变化也有代价，没有以单次测量归因。

未达到既有连续batch两尺度20%门槛。生产撤回该Cast，只保留[研究补丁](../debug/fp16_padded_batch_cast.patch)、[固定基线复现脚本](../tools/run_fp16_padded_cast_probe.sh)和[原始构建/运行证据](fp16-padded-cast-evidence-2026-10-11/model-summary.json)。无须为本实验再次运行Judge，#66保持开放并标记研究停止。
