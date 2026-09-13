# row-batch4 production 实验

日期：2026-09-13

基线：`bulk-prefix-pack`。分支：`row-batch4-production`。

## 目的

把 profiling 中已经归因确认的 B=4 跨行 batching 直接送入真实 CANNJudge，不再增加新的 simulator 性能门槛。

第二轮归因结果：

- 通用逐行：6.170 us；
- 固定形状逐行：6.020 us；
- 固定形状 batch4：4.980 us。

因此专门化单独约 2.4%，batching 独立约 17.3%。

## production 范围

只启用 FP32 `E=1024,k=8` 的四行 batch。`kernel.asc` 的 host 调度保持不变，仍传入 `blockNum=min(rows,coreNum)`。

Device dispatcher 仅当以下条件同时满足时走 batch4：

- `E=1024`；
- `k=8`；
- `rows` 可被 4 整除；
- `rows >= 4*blockNum`。

最后一条保证四行分组后 `rows/4 >= blockNum`，不会因为 batching 让已启动的 block 无工作。其他形状继续走原 dense-repeat / prefix-merge 路径。

batch kernel 沿用 profiling 已通过 correctness 的数据流：一次 `Sort32` 处理 128 repeats，前两层 `MrgSort` 跨四行批量，final 2-way merge / GatherMask / transform 逐行。

## 提交文件

华为平台空工程中使用 `row-batch4-production` 分支：

- `submission/kernel.asc` -> 平台 `kernel.asc`；
- `submission/dense_repeat_override.h` -> 同目录 `dense_repeat_override.h`。

本分支实际 production 改动集中在 header；仍建议两份一起复制，避免和旧空工程内容混用。

## 提交前最小验证

只要求：结构检查通过、CANN 编译通过、`rows=4/8,E=1024,k=8` production probe 正确。之后直接提交 Judge，不再先扩展 E/B 或增加性能验证轮次。
