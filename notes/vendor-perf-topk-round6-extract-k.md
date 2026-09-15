# Vendor Perf TopK 第六轮：按真实 k 提取并批量后处理

日期：2026-09-15

## 出发点

第五轮把后处理从逐行 `TransformSelected(k<=8)` 改成跨行一次 `TransformSelected(validRows * 8)` 后，真实 Judge 的主要受影响点出现明确下降：

```text
case 3 : 154.49 -> 139.62 us   -9.6%
case 7 :  29.97 ->  18.69 us  -37.6%
case 9 :  46.68 ->  28.38 us  -39.2%
case 13: 273.36 -> 172.63 us  -36.8%
```

随后两次 A/B 已经把收益来源基本拆清：

1. 只撤销 `k=8` 的 batch copyout，结果几乎不变；
2. 保留 packed layout 和 copyout，只把 transform 恢复为逐行 8 元素调用，结果几乎退回第四轮：

```text
case 3 : 156.20 us
case 7 :  27.67 us
case 9 :  43.05 us
case 13: 268.87 us
```

因此第五轮的主要收益来自 **减少大量短 transcendental vector 调用，将后处理合并为一个更长的 batched transform**。

## Huawei Perf 中的对应结构

继续检查 `hicann/ops-transformer` 的 `MoeGatingTopKSoftmaxPerf` 后发现，vendor 路径并不会固定抽取每行 top-8 再处理全部 8 个值。

host 侧通过 `GenValuesMask(k)` 生成交错 value mask；kernel 的 `ExtractKFP32Perf` 直接用 pattern `GatherMask` 从最终 pair 布局中抽取每行真实的前 k 个 value，并压成连续的 `[curRowsNum, k]`。随后 CopyOut 也直接按 `curRowsNum * k` 连续写回。

这与本题第五轮已经验证有效的 batched transform 可以直接组合。

## 第六轮 bounded change

selection / merge / staging / batchRows / dispatch 全部保持第五轮不变，只替换最终 value extraction 与后处理布局：

```text
第五轮：
final pairs
  -> GatherMask 每行 32 values
  -> PackTop8Values -> [validRows, 8]
  -> TransformSelected(validRows * 8)
  -> k=8 连续 copyout，k<8 逐行 copyout

第六轮：
final pairs
  -> pattern GatherMask(valueMask(k)) -> [validRows, k]
  -> TransformSelected(validRows * k)
  -> 一次连续 copyout(validRows * k)
```

具体实现：

- `BuildVendorValueMask(k)` 生成 `01 01 ...` 的 value mask；
- 每 core 在进入 batch loop 前初始化一次 pattern UB；
- `ExtractKValues()` 参考 Huawei `ExtractKFP32Perf` 使用 `GatherMaskParams`，`repeatTimes=validRows`，每次跨过一个 final-pair row；
- 删除第五轮的 `gatherBuffer_` 与 `PackTop8Values()`；
- output/tmp 仍按最多 `batchRows * 8` 分配，避免额外改变 UB/batch admission；
- transform count 改为 `validRows * k_`；
- output 已经连续压紧，因此所有 `k=2..8` 都走一次连续 `DataCopyPad`。

## 验证原则

- `tools/check_vendor_perf_topk.py` 更新为检查 true-k pattern extraction；
- 现有 production probe 已覆盖 `k={2,3,4,5,7,8}`、非 32 对齐 E、多核与尾 batch，足以直接检验 compaction correctness；
- 先跑结构 checker，再在真实 Bisheng/CANN 环境编译并跑 production probe；
- correctness 通过后直接完整 15-point Judge；
- 不增加 simulator 性能准入。

## 本地验证

2026-09-15 本地验证已完成：

- `python3 tools/check_vendor_perf_topk.py`：PASS；
- Bisheng/CANN CPU probe 成功编译；
- `topk_prefix_production_probe`：全部 PASS。

因此第六轮已经通过结构检查与本地 correctness 门槛。

## 真实 Judge

15/15 Pass：

```text
case 1 :    2.33 us
case 2 :    3.77 us
case 3 :  139.80 us
case 4 :    7.80 us
case 5 :   16.63 us
case 6 :   23.95 us
case 7 :   17.73 us
case 8 :   86.99 us
case 9 :   27.71 us
case 10:  244.10 us
case 11:  252.30 us
case 12:  650.44 us
case 13:  176.97 us
case 14:    2.13 ms
case 15:    3.46 ms
```

与第五轮直接比较，最关键的 3/7/9/13 为：

```text
case 3 : 139.62 -> 139.80 us   +0.13%
case 7 :  18.69 ->  17.73 us   -5.1%
case 9 :  28.38 ->  27.71 us   -2.4%
case 13: 172.63 -> 176.97 us   +2.5%
```

第六轮没有再次出现第五轮 30--40% 级的结构收益。结合此前 A/B，可以把机理进一步区分：

- 把“每行一次很短的 `TransformSelected(8)`”合并成“每 batch 一次较长 transform”非常重要；
- 在已经只发射一次 transform 后，把元素数从 `validRows*8` 降到 `validRows*k`，真实壁钟只出现小幅或噪声级变化；
- 因此当前 transform 成本里，固定的向量指令/同步发射成本比少处理几个元素更值得关注。

第六轮的 true-k extraction 可以保留为更贴近 vendor、布局也更干净的实现，但不再把“减少 transform 元素数”作为主要优化方向。下一步优先检查 `StableSoftplusSqrt` 自身的指令与 barrier 成本；如果无法进一步压缩，再回到 vendor selection 路径寻找剩余结构差距。
