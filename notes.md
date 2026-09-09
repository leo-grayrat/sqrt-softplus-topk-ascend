# Notes

## 已确认

- `sqrt(softplus(x))` 对 logits 单调递增，因此可以先取 logits Top-K，再只计算选中的分数。
- 题面 PyTorch 示例对 FP16 输入实际返回 FP32，但表格又要求输出 dtype 与输入一致；已在 reference 中分开表达这两层含义。
- 当前尚未拿到本题 CANNJudge starter project，因此比赛实际 CANN 版本、SoC 和工程结构仍未知。

## 实验记录格式

每次真实实现或性能实验只记这些：

```text
日期：
commit：
改动：
环境：
验证：
结果：
结论：保留 / 放弃 / 待查
```

没有真实 NPU/Judge 数据时，不写“性能提升”。
