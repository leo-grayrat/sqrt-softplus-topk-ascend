# FP32 case 15 scratch 复用：约 1.3% 模型收益，停止（#63）

从 PR #62 后 main 开始，case 15 为 FP32、T65555/E1024/k64，最新 Judge 为 1.65ms。本轮只有本地证据，没有提交新 Judge。**批次扩大没有形成足够收益，生产 large-k 已恢复 main；研究补丁仅存 debug。**

## 假设与实现

限定 k64、padded inner1024：Sort32 前 pairB 前半部容纳需要重排的填充输入，后半部容纳每批刷新的零 index；Sort32 屏障后合并覆盖 pairB。删除独立 input/index 两份整行缓冲，batch 4→7，Sort32 最多224 repeats。8行需256 repeats，超过上限，未采用。

E1024 显式 UB 从133152 B（4行）变为175648 B（7行），含输入双缓冲、两份 pair、输出/临时区和pattern，不含框架开销。三层合并树和每行合并元素量没有变化，因此调用次数减少不能直接推导吞吐收益。

## 同源完整模型

CANN9.0 / dav-2201，Ascend910B1 simulator，单核，相同 run_kernel host入口、生产dispatch、输入和核数。包含输入、selection、变换和写回；以下为单次 duration_time，不是真机/Judge性能。

| E1024/k64 行数 | 阶段开始 main/us | scratch候选/us | 耗时减少 |
|---|---:|---:|---:|
| 28 | 39.82 | 39.32 | 1.26% |
| 56 | 77.54 | 76.54 | 1.29% |
| 1 | 3.66 | 3.65 | 0.27% |

28/56行覆盖候选4/8个连续batch与main7/14个batch。两组均未达20%准入门槛，不要求用户为此跑Judge。

| 28行指令调用数 | main | 候选 |
|---|---:|---:|
| VBS32 | 7 | 4 |
| VMS4V2 | 21 | 12 |
| MOV_UB_TO_UB | 21 | 12 |
| BAR | 137 | 86 |

调用减少而总元素量仍相同，完整模型仅改善约1.3%。这些计数不等于元素量或速度提升倍数，不对各指令累计时间求和解释wall time。

## 验证和复现

- main/候选实际 run_kernel 入口 CPU 各46/46，含1/6/7/8/13/14/15/28/29/56/57行边界、E993/1000/1001/1008/1016/1017/1023/1024、多核尾批次、k32原路径；输入含重复值、全-inf、末段/单段极值及±10000。
- baseline4次、候选10次 simulator 输出检查通过；候选包含相邻 E992/k64、E1024/k63 的通用TopK回退。这两条不用CPU检查：其未使用的空srcIndexLocal受已有CPU检查器限制，见syntax-errors第9条。
- 候选完整 submission dav-2201 编译通过。两次完整测量脚本均成功结束。源代码与基线/研究补丁构建副本核对一致。
- 首次改为host入口时忘记初始化平台，双方第一个E97/k32用例输出全零并失败；按现有probe加入PlatformAscendCManager初始化后重新完整构建/验证，不使用失败运行的性能数据。不是生产selection错误。
- [研究补丁](../debug/fp32_large_scratch_reuse.patch)、[同源生产入口probe](../debug/cpu_topk_probe/topk_fp32_large_production_probe.asc)、[复现脚本](../tools/run_fp32_large_scratch_probe.sh)、[日志/模型/指令计数](fp32-large-scratch-evidence-2026-10-11)。脚本固定本阶段开始main，candidate只应用研究补丁，后续生产变化不会混入实验。

#63记录该方向停止；#52极大输入数值边界仍独立。随后case5紧凑合并另以#64跟踪，交付文件不包含本轮large-k改动。
