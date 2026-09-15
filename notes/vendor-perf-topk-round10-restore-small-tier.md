# Vendor Perf TopK 第十轮：恢复 small-inner 的 16-row tier

日期：2026-09-15

## 出发点

Round 8 把固定 `16/8/4` batch 改成 `5120 / inner` 后，真实 Judge 出现分化：case 3/9/13 有收益，但 case 7 从 17.73 us 退化到 20.65 us。

Round 9 在 Round 8 上增加全局 32-row cap，结果与 Round 8 几乎完全重合：

- case 3：132.97 -> 132.89 us；
- case 7：20.65 -> 20.50 us；
- case 9：27.43 -> 27.30 us；
- case 13：172.43 -> 172.59 us。

因此可以排除“隐藏 case 的关键退化来自 batchRows>32”这一简单解释。case 7 的实际 batch 可能在 17--32 行，或者属于 `inner>=160`、Round 9 没有改变的档位。

## bounded change

基线：`vendor-perf-topk-round8-larger-batch`。

只修改 batch capacity：

```cpp
if (inner <= 256) {
    return 16;
}
capacity = 5120 / inner;
```

因此：

```text
inner<=256 : 恢复 Round 6 的 16 rows
inner=288  : Round 8 的 17 rows
inner=320  : Round 8 的 16 rows
inner=512  : Round 8 的 10 rows
inner=1024 : Round 8 的 5 rows
```

selection / merge / true-k GatherMask / batched transform / copyout 均不变。

## 这轮要回答的问题

这轮不是继续找一个全局最佳 row cap，而是做档位定位：

- 如果 case 7 回到约 17.7 us，而 case 3/9/13 中只有某些点回退，就能判断哪些隐藏 case 位于 `inner<=256`；
- 如果 case 7 仍在约 20.5 us，则它的 Round 8 退化来自 `inner>256` 档位，下一轮再单独恢复 `257<=inner<=512` 的旧 8-row tier；
- 如果 case 3 从约 133 us 回到约 140 us，则 case 3 的 Round 8 收益来自 small-inner batch 扩大；若保持不变，则其收益来自更大的 inner tier。

## correctness 与资源边界

Round 8 probe 已包含多组会直接命中该策略的 case，例如：

```text
rows=81 E=64  k=8 cores=1
rows=41 E=128 k=5 cores=1
rows=21 E=256 k=4 cores=1
```

前两组从 Round 8 的大 batch 恢复为 16-row 分组；`E=256` 也从 20 行恢复为 16 行。

资源边界仍安全：`inner<=256` 时 batchElements<=4096；`inner>256` 时仍遵守 Round 8 的 5120-element budget。最大 `Sort32` repeat 仍不超过 160。
