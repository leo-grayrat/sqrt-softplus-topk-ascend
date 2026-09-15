# Vendor Perf TopK 第八轮：扩大 row batch

日期：2026-09-15

## 背景

第五轮已经证明：把逐行短向量后处理合并成跨行 batched transform，可以在真实 Judge 上获得 30--40% 级收益。第六轮进一步说明，在已经只调用一次 transform 的前提下，单纯减少一次调用中的元素数收益很小。第七轮 auto-sync A/B 又确认，BiSheng 会把 transform 内部的 PIPE_V 同步逐个重新插回，因此不能靠删手写 barrier 获得进一步重叠。

因此下一步回到 row batching 本身：如果一个 core 仍被拆成多个较小 group，那么 Sort32 / merge / GatherMask / transform 这一整串固定调用仍会重复发射。

## 与 Huawei Perf 的差异

Huawei `MoeGatingTopKSoftmaxPerf` 的 host tiling 并没有把 batchRows 固定成 16/8/4，而是通过 `calcMaxRowInUb()` 根据：

- 实际 UB 大小；
- 对齐后的 col；
- 每核行数；
- 各个通用 buffer 的总占用；

动态求能够放进 UB 的最大 `curRowInUb`。

当前本仓库第六轮仍使用：

```text
inner <= 256  -> 16 rows
inner <= 512  ->  8 rows
else          ->  4 rows
```

这组上限最初主要是把一次 `Sort32` repeat 控在约 128，但官方 `Sort32` 的 repeatTimes 实际允许到 255，因此仍有空间。

## 第八轮 bounded change

基线：`vendor-perf-topk-round6-extract-k`。

selection / merge / true-k extraction / batched transform / copyout 全部不变，只修改 `SelectVendorBatchCapacity()`。

引入：

```cpp
constexpr int32_t VENDOR_BATCH_ELEMENT_BUDGET = 5120;
capacity = VENDOR_BATCH_ELEMENT_BUDGET / inner;
```

并把 capacity 限制在 `[1,255]`。

典型 capacity：

```text
inner=64   -> 80 rows
inner=96   -> 53 rows
inner=128  -> 40 rows
inner=256  -> 20 rows
inner=288  -> 17 rows
inner=512  -> 10 rows
inner=544  ->  9 rows
inner=1024 ->  5 rows
```

## 安全边界

当前 vendor kernel 的 UB 分配最坏可按每 batch 估算为：

```text
2 * input staging queue + inputBuffer + indexBuffer
+ pairA + pairB + outputQueue + floatTmp + pattern

<= 32 * (batchRows * inner) + 64 * batchRows + 32 bytes
```

本题 `E>=64`，因此 `inner>=64`。在 `batchRows * inner <= 5120` 下：

```text
batchRows <= 5120 / 64 = 80
max UB <= 32*5120 + 64*80 + 32
       = 168992 bytes
```

A2/A3 Unified Buffer 总大小为 192 KiB，因此仍保留约 27 KiB 余量。

同时：

```text
Sort32 repeatTimes <= 5120 / 32 = 160 < 255
GatherMask repeatTimes <= 80
```

中间 `MrgSort` 的最大 repeat 也低于 255。

所以这轮不是无界扩大 batch，而是在现有 buffer 布局和硬件 repeat 范围内提高批量度。

## production probe 补充

原 224-case probe 大多数单核测试只到 rows=16，多核测试的 rows/core 也通常没有跨过旧 16/8/4 上限，因此不能证明新 batch 真正被执行。

第八轮增加专门的大 batch correctness case：

```text
rows=81 E=64   k=8 cores=1
rows=54 E=65   k=7 cores=1
rows=41 E=128  k=5 cores=1
rows=21 E=256  k=4 cores=1
rows=18 E=257  k=8 cores=1
rows=11 E=512  k=3 cores=1
rows=10 E=513  k=5 cores=1
rows=6  E=1024 k=8 cores=1
```

这些 case 都故意超过旧 capacity，并且多为 `new_capacity + 1`，同时验证最大 batch 和尾 group。

## 预期解释

如果真实 Judge 的 3/7/9/13 等点进一步下降，说明剩余差距中仍有一部分来自每核 group 过碎、固定 API 序列重复发射。

如果基本持平，则说明第五轮 batched transform 已经跨过主要的发射开销阈值，继续扩大 row batch 不再重要；下一步应回到 selection/merge 本体或 host block 调度。

如果本地编译出现 UB overflow，则说明 5120 元素预算仍过大，需要根据实际 pipe 分配把预算下调，而不是改变算法结构。
