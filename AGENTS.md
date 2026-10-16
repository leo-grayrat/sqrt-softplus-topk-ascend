# AGENTS.md

这个仓库用于 CANN 挑战赛 SqrtSoftplusTopK 算子实验。进入仓库后先读 `README.md`、`notes.md` 和 `notes/syntax-errors.md`。

## 约束

1. **不要猜 CANN/Ascend C API。** 只使用与比赛环境匹配的官方文档、官方 sample 或 CANNJudge starter project 中已经出现的接口。
2. **不要自己造比赛工程。** `submission/` 以 CANNJudge 下载的真实 starter project 为准；在拿到它之前不要臆造 `op_host/`、`op_kernel/` 等结构。
3. **不要混淆验证层级。** 本地 Python 测试、CANN 编译、真实 NPU 运行、Judge AC、性能提升是不同结论；只能报告已经实际验证到的那一层。
4. **性能改动必须有真实比较。** 没有 NPU/Judge 数据时只能称为候选优化；失败方案也写入 `notes.md`，避免以后重复尝试。
5. **每个明确问题都用 GitHub issue 跟踪。** 发现问题后先建 issue；修复提交后保持打开，只有用户在本地或 CANNJudge 重新验证该问题不再出现后才关闭。Bisheng/CANN 的可复用编译坑同时简记到 `notes/syntax-errors.md`。

## Reference

`reference/reference.py` 中的 `official` 用来保留题面 PyTorch 示例的计算行为，不要为了方便随意改写它。若题面、starter project 与 Judge 的 contract 发生冲突，先记录冲突，再依据真实比赛环境处理。

## CANN 资料

[已核实的 CANNBot Skills 与 asc-devkit 资料入口](notes/cannbot-resources-2026-10-16.md)。新增 API 用法先查匹配版本文档的所有重载、芯片支持和参数单位，再以本机 SDK、编译和输出检查验证。CANNBot 的默认 A3/CANN8.5 环境与本仓 CANN9/dav-2201/910B 不同，不能照搬；950/arch35 专用实现也不能直接用于当前硬件。
