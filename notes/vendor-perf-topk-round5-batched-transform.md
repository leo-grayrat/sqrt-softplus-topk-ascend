# Vendor Perf TopK 第五轮：批量后处理

日期：2026-09-14

## 第四轮 Judge 结果

第四轮在第三轮 padded-inner vendor-perf 基础上加入：

- 一个 batch 一次 `DataCopyPad` 搬入 32B 对齐 staging；
- 非 32-element 对齐时再批量 rearrange 到 `inner=ceil(E/32)*32`；
- Sort32 / prefix copy / MrgSort / GatherMask 只处理真实 `validRows`。

真实 Judge 15/15 正确：

```text
case 3 : 154.49 us
case 7 :  29.97 us
case 9 :  46.68 us
case 13: 273.36 us
```

相对第三轮约为：

```text
case 3 : +1.8%
case 7 : -2.0%
case 9 : -4.1%
case 13: -4.3%
```

结论：输入 staging 与真实 `validRows` 值得保留，但只贡献小幅收益，不再继续深挖该层。

## 第五轮目标

保持 selection / merge / staging / dispatch 不变，只优化 `GatherMask` 之后的本题特有后处理。

此前每行都会单独执行：

```text
TransformSelected(k<=8)
  Abs -> Muls -> Exp -> Adds -> Ln -> Maxs -> Add -> Sqrt
```

因此一个 batch 会反复发射多组非常短的 transcendental vector 操作和 barrier。

第五轮改成：

1. `GatherMask` 仍按每行 32 values 写入 wide scratch；
2. 用一次跨行 UB `DataCopy` 取每行前 8 个 FP32，压紧成 `[validRows, 8]`；
3. 一次 `TransformSelected(validRows * 8)` 完成整个 batch 的 `sqrt(softplus)`；
4. output queue 与 transform tmp 都缩成 `batchRows * 8` floats。

## CopyOut 约束

CANN `DataCopyPad` 的 `DataCopyExtParams` 中，如果源操作数位于 VECIN/VECOUT，`srcStride` 单位是 32B data block，而不是 byte。

因此对于 `k=2..7`，一行有效输出只有 `k*4` bytes，无法用一个整数 `srcStride` 从 8-wide（32B）packed rows 中精确跨行抽取。

为了保持 exact correctness，本轮不额外引入新的 compaction 算法：

- `k=8`：8 个 FP32 恰好 32B，整 batch 连续一次 copyout；
- `k<8`：保留逐行 `DataCopyPad` copyout。

所以本轮核心验证的是 **batched transcendental transform**，不是强行消灭所有 copyout call。

## 验证原则

- 结构 checker 要求 top-8 跨行 pack 与一次 `validRows*8` transform；
- CPU production probe 继续覆盖 k={2,3,4,5,7,8}、非 32 对齐 E、多核与尾 batch；
- 不增加 simulator 性能准入；
- CPU correctness 通过后直接完整 Judge。

## 第五轮真实 Judge

production probe 通过后直接提交完整 Judge，15/15 Pass：

```text
case 1 :   2.26 us
case 2 :   4.12 us
case 3 : 139.62 us
case 4 :   8.60 us
case 5 :  17.34 us
case 6 :  24.76 us
case 7 :  18.69 us
case 8 :  87.63 us
case 9 :  28.38 us
case 10: 244.91 us
case 11: 253.24 us
case 12: 650.67 us
case 13: 172.63 us
case 14:   2.13 ms
case 15:   3.46 ms
```

相对第四轮重点 case：

```text
case 3 : 154.49 -> 139.62 us   -9.6%
case 7 :  29.97 ->  18.69 us  -37.6%
case 9 :  46.68 ->  28.38 us  -39.2%
case 13: 273.36 -> 172.63 us  -36.8%
```

这是 vendor-perf 路径中又一次结构级真实收益，尤其 7/9/13 的降幅远超 Judge 噪声。

## A/B 1：撤销 k=8 整批 copyout

分支：`vendor-perf-topk-ab-row-copyout`

只把 `k=8` 的连续整批 copyout 恢复成逐行 `DataCopyPad`，保留第五轮 batched transform 与 packed layout。

真实 Judge 15/15 Pass：

```text
case 3 : 140.21 us
case 7 :  18.70 us
case 9 :  28.04 us
case 13: 177.88 us
```

相对第五轮分别约为 `+0.4% / +0.05% / -1.2% / +3.0%`。

结论：`k=8` 连续整批 copyout 不是第五轮大收益来源；不值得继续围绕 copyout 做复杂压缩或专门优化。

## A/B 2：撤销 batched TransformSelected

分支：`vendor-perf-topk-ab-row-transform`

保留第五轮的：

- `PackTop8Values`；
- `[validRows, 8]` packed layout；
- output buffer 与 copyout；
- selection / merge / staging / dispatch。

唯一把一次：

```text
TransformSelected(validRows * 8)
```

恢复成：

```text
for each row:
    TransformSelected(8)
```

真实 Judge 15/15 Pass：

```text
case 3 : 156.20 us
case 7 :  27.67 us
case 9 :  43.05 us
case 13: 268.87 us
```

这相对第五轮分别退化约：

```text
case 3 : +11.9%
case 7 : +48.0%
case 9 : +51.7%
case 13: +55.8%
```

并且已经接近第四轮：

```text
case 3 : 156.20 vs 154.49 us
case 7 :  27.67 vs  29.97 us
case 9 :  43.05 vs  46.68 us
case 13: 268.87 vs 273.36 us
```

其中 case 7/9 仍比第四轮约快 7--8%，说明 packed 后处理布局本身可能还有少量贡献；但第五轮 30--40% 级的大头已经可以归因于 **把许多 8-element transcendental vector 操作合并成一次较长的 batched transform**。

## 当前结论

1. 第五轮成功，production 应保留 batched `TransformSelected(validRows * 8)`。
2. batch copyout 的贡献接近噪声，降级，不再深挖。
3. 本题后处理里的短 `Abs/Muls/Exp/Adds/Ln/Maxs/Add/Sqrt` 调用具有很高的固定发射/同步成本；跨行合并后真实 Judge 可获得 30--40% 级收益。
4. 下一阶段应优先检查还有哪些逐行、短长度 vector 操作可以跨行合并，或能否减少 `TransformSelected` 实际处理的无用 top-8 元素，而不是继续围绕 GM copyout 做文章。
