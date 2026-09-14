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
