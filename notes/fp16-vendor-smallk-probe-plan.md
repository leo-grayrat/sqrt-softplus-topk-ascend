# FP16 vendor-perf small-k probe

## 已确认的 vendor 行为

对 Huawei `MoeGatingTopKSoftmaxPerf<T>` 的源码复核显示：`T != float` 时并不是让 half/bfloat16 原生进入 `Sort32/MrgSort`，而是先在 UB 中 Cast 到 FP32，再调用 `TopKFP32PerfPhase1`，最终按输出 dtype Cast 回去。

因此本实验不再验证“half pair 原生布局”，而直接验证与 vendor 一致的路径：

```text
GM half -> UB half -> Cast FP32 -> Sort32<float>
-> top-8 prefix -> MrgSort<float> -> GatherMask values
-> Cast half -> GM half
```

首个目标 shape 固定为 Judge case 12 的 `E=256,k=3`。probe 不修改 production dispatch，不追求性能，只检查：

1. half -> FP32 Cast 后进入现有 FP32 pair layout 是否正确；
2. 每个 32 元素块截取 top-8 prefix 后，MrgSort 能否得到全局 top-k；
3. GatherMask value pattern 能否抽出真实 k 个 FP32 value；
4. Cast 回 half 后与 host golden top-k score 一致。

如果 probe 通过，下一阶段再把 production vendor path 扩到 FP16 small-k；若失败，按实际 CANN checker / Bisheng 报错定位具体 API/layout 约束，不绕开错误直接改 production。
