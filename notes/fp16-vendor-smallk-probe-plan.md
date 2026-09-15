# FP16 vendor-perf small-k probe

本实验只回答一个问题：dav-2201 / CANN 9.0 下，half 输入能否原生沿用当前 FP32 vendor-perf 的 `Sort32 -> prefix MrgSort -> GatherMask` 执行结构与 pair 布局。

首个目标 shape 固定为接近 Judge case 12 的 `E=256,k=3`。probe 不修改 production dispatch，不追求性能，只检查：

1. half `Sort32` 输出 pair 的 value/index 解释是否和预期一致；
2. 每个 32 元素块截取 top-8 prefix 后，half `MrgSort` 是否能正确得到全局 top-k；
3. `GatherMask` 能否按 value pattern 抽出真实 k 个 half value；
4. 输出与 host golden top-k score 一致。

若 half 原生 pair 路径成立，下一步再模板化 production vendor kernel；若 API/layout 不成立，再考虑 GM half -> UB cast float -> 复用 FP32 selection。
