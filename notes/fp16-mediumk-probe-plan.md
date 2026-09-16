# FP16 medium-k keep-16 probe

目标：在接入 production 前，直接用 CANN CPU twin-debug 验证 `9<=k<=16` 的 keep-16 selection 布局。

当前 probe：`debug/cpu_topk_probe/topk_fp16_vendor_medium_probe.asc`。

执行结构：half GM -> FP32 Cast -> Sort32 -> 每 32 元素保留 top-16 pair -> MrgSort 分层归并 -> GatherMask 取真实 k -> FP32 sqrt(softplus) -> half。

覆盖：
- E=128,k=16：case 10 的核心结构；
- E=200,k=16：case 5，含 half 行对齐后再重排；
- E=512,k=10：case 14 的核心结构；
- 另含多个 padding / k 边界组合。

机械字符串 checker 已删除。本实验只把能实际执行 selection/layout 的 CPU probe、Bisheng/device 编译和后续真实 Judge 作为有效验证。
