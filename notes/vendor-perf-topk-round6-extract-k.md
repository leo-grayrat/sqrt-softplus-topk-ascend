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

因此第六轮已经通过结构检查与本地 correctness 门槛。下一步直接运行完整 15-point Judge，不再插入 simulator 性能测试。
