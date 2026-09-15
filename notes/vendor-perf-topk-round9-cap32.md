# Vendor Perf TopK 第九轮：5120-element budget + 32-row cap

日期：2026-09-15

## 出发点

第八轮把 row batch 从固定 `16/8/4` 改成 `5120 / inner` 后，真实 Judge 出现明显的非单调结果：

- case 3：139.80 -> 132.97 us，约 -4.89%；
- case 7：17.73 -> 20.65 us，约 +16.47%；
- case 9：27.71 -> 27.43 us，约 -1.01%；
- case 13：176.97 -> 172.43 us，约 -2.57%。

15 点总时间约 +0.32%，因此 Round 8 不能直接作为新生产基线。

第八轮对不同 inner 的改变量差异很大：

```text
inner=64:   16 -> 80
inner=128:  16 -> 40
inner=256:  16 -> 20
inner=512:   8 -> 10
inner=1024:  4 -> 5
```

因此 case 7 的退化更可能来自 small-inner 的过度 batching，而不是 5120-element budget 这个总体方向本身。

## bounded change

基线：`vendor-perf-topk-round8-larger-batch`。

保留：

```cpp
VENDOR_BATCH_ELEMENT_BUDGET = 5120;
```

新增：

```cpp
VENDOR_MAX_BATCH_ROWS = 32;
```

实际 capacity：

```cpp
capacity = min(5120 / inner, 32);
```

因此典型值变成：

```text
inner=64    -> 32  (Round 8: 80)
inner=96    -> 32  (Round 8: 53)
inner=128   -> 32  (Round 8: 40)
inner=160   -> 32  (Round 8: 32)
inner=256   -> 20  (不变)
inner=512   -> 10  (不变)
inner=1024  ->  5  (不变)
```

selection / merge / true-k extraction / transform / copyout 均不变。

## 为什么这一轮比直接换 budget 更干净

如果直接把 5120 改成 4096 或更低，中大 inner 的 capacity 也会同时变化，无法判断 case 7 的退化究竟来自：

- 小 inner 下 row 数过多；还是
- 所有 shape 的 batch 都整体偏大。

32-row cap 只影响 `inner < 160`，因此可以单独回答 small-inner over-batching 是否存在。

## 安全边界

仍有：

```text
batchRows * inner <= 5120
batchRows <= 32
Sort32 repeats <= 160
```

当前 UB 上界可保守估算为：

```text
32 * 5120 + 64 * 32 + 32 = 165920 B
```

低于 192 KiB。

## correctness 覆盖

沿用 Round 8 已增加的大 batch probe，其中以下 case 会直接命中新的 32-row cap，并覆盖多 group/tail：

```text
rows=81 E=64  k=8 cores=1   # 32 + 32 + 17
rows=54 E=65  k=7 cores=1   # 32 + 22
rows=41 E=128 k=5 cores=1   # 32 + 9
```

中大 inner 的 Round 8 probe 继续验证未受影响的 20/10/5-row capacity。

## 真实 Judge 结果

15/15 Pass，0% 输出错误。关键点：

| case | Round 8 | Round 9 | 变化 |
|---|---:|---:|---:|
| 3 | 132.97 us | 132.89 us | -0.06% |
| 7 | 20.65 us | 20.50 us | -0.73% |
| 9 | 27.43 us | 27.30 us | -0.47% |
| 13 | 172.43 us | 172.59 us | +0.09% |

15 点绝对时间总和 Round 8 约 7263.56 us，Round 9 约 7263.08 us，差异约 -0.007%，可视为完全持平。

## 结论

32-row cap 对隐藏 Judge 基本没有影响，因此 case 7 的 Round 8 退化不能归因于 `batchRows > 32`。更准确地说，相关 case 的实际 batch 可能本来就在 17--32 行区间，或者落在 `inner>=160`、本轮没有改变的档位。

下一轮不再继续改全局 row cap，而是做档位隔离：只把 `inner<=256` 恢复到 Round 6 的 16 行，`inner>256` 保持 Round 8 的 `5120/inner`。这样可以直接判断 case 7 是否来自第一档，同时保留中大 inner 的 Round 8 行为。
