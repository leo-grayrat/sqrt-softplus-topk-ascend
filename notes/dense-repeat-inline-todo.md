# Dense-repeat production 提交形态待整理

当前 `submission/kernel.asc` 通过 `#include "dense_repeat_override.inc"` 引入 dense-repeat 实现；后续修复也发生在 `.inc` 文件中，因此 `kernel.asc` 本体在 2026-09-12 17:00 左右后没有新的实现提交。

本地 CPU production probe 会从完整 `submission/` 目录解析该 include，因此 66/66 PASS 只证明“完整目录编译”正确，不足以证明 CANNJudge 若只采集 `kernel.asc` 时仍可提交。

在下一次 CANNJudge 前，需要把 dense-repeat 实现直接并回 `kernel.asc`，去掉对旁挂 `.inc` 的依赖，再做一次 clean rebuild + production probe。
