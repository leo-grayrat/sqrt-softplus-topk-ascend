# FP16 vendor-perf small-k probe

本实验回答：dav-2201 / CANN 9.0 下，FP16 输入能否沿用 Huawei vendor-perf 的 small-k selection 结构。

源码复核后确认，Huawei `MoeGatingTopKSoftmaxPerf<T>` 对非 FP32 输入并不是直接做 half `Sort32`，而是先在 UB 中 `Cast` 到 FP32，随后统一进入 FP32 `Sort32 -> prefix MrgSort -> GatherMask`，最后再 Cast 回原 dtype。因此本仓库采用同样路线。

首个 probe 固定为接近 Judge case 12 的 `E=256,k=3`：

1. GM/UB half 输入；
2. 批量 Cast 到 FP32；
3. FP32 Sort32；
4. 每个 32 元素块保留 top-8 prefix；
5. MrgSort 得到最终前缀；
6. GatherMask 抽真实 k 个 value；
7. Cast 回 half，与 host golden 比较。

## 结果

用户本地 CANN 9.0 / dav-2201 CPU twin-debug 已完成 checker、编译和运行，probe PASS。

因此 feasibility 阶段结束：half 输入可以通过 `half -> FP32 vendor selection -> half` 结构正确完成 small-k selection。下一阶段进入 production 集成，并保持 FP32 既有路径不动。
