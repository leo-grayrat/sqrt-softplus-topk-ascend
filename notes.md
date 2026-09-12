# Notes

## 已确认

- `sqrt(softplus(x))` 对 logits 单调递增，因此可以先取 logits Top-K，再只计算选中的分数。
- 题面 PyTorch 示例对 FP16 输入实际返回 FP32，但表格又要求输出 dtype 与输入一致；已在 reference 中分开表达这两层含义。
- 已拿到本题 CANNJudge starter project；模板默认目标架构为 `dav-2201`。
- 当前实现已在 CANNJudge 通过全部正确性 case，首个性能基线得分 22.80，排名 63；详细时间见 `notes/perf-experiments.md`。
- 本地环境已确认可使用 CANN 9.0 / Bisheng 编译当前工程。

## 调试与实验约定

- 发现明确问题先建 GitHub issue；修复后必须等本地或 CANNJudge 复验通过再关闭。
- CANN / Bisheng 语法与编译坑记录在 `notes/syntax-errors.md`。
- 性能实验记录在 `notes/perf-experiments.md`；没有真实 NPU/Judge 数据时，不写“性能提升”。
- profiling / simulator 的构建、型号映射与已踩坑记录在 `notes/profile-topk-cost-model.md`；对应进行中的成本模型任务是 issue #31。
- CPU twin-debug 只能证明 API / layout / correctness，不等价于 simulator/NPU kernel launch，也不作为性能证据。
