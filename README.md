# SqrtSoftplusTopK Ascend

2026 CANN 挑战赛初赛 `SqrtSoftplusTopK` 算子的实验仓库。

目前只完成 **PyTorch 正确性参考与仓库骨架**，尚未接入比赛的 CANNJudge starter project，也没有声称任何 Ascend C 实现已经编译、运行、AC 或获得性能提升。

## 题目

输入 `router_logits`，形状为 `(T, E)`，对每个元素计算：

```text
sqrt(softplus(x))
```

然后逐 token 取最大的 `k` 个分数。题目范围包括 FP16/FP32、`E=64..1024`、`k=1..64`。

## 一个直接可用的化简

`sqrt(softplus(x))` 单调递增，因此 Top-K 的专家不会因为这个变换改变：

```text
TopK(sqrt(softplus(x)))
=
sqrt(softplus(TopK(x)))
```

所以候选实现可以先在原始 logits 上取 Top-K，再只对选中的 `T*k` 个值做 FP32 softplus 和 sqrt。

`reference/reference.py` 中同时保留：

- `official`：照题面 PyTorch 示例的计算过程；
- `topk_first`：先 Top-K 再变换的等价写法；
- `expected_output`：按题面“输出 dtype 与输入一致”的要求转回 dtype。

## 题面中的 dtype 矛盾

题面表格要求输出 dtype 与输入一致，但给出的 PyTorch 示例先执行 `router_logits.float()`，随后直接返回，因此 FP16 输入在示例代码中实际得到 FP32 输出。

仓库不偷偷忽略这个矛盾：`official` 保留示例行为，`expected_output` 单独表达题面声明的输出类型要求。真正 Ascend C 工程接入后，以 starter project / Judge 的实际 contract 再确认。

## 测试

```bash
python -m pytest tests/test_reference.py -q
```

当前测试覆盖 FP16/FP32、`k=1/6/64`、2 的幂与非 2 的幂 `E`、20 附近、极值、`-inf`、重复值和固定随机输入。

## 下一步

1. 从 CANNJudge 下载本题真实 starter project，放入 `submission/`；
2. 确认比赛实际 CANN 版本与 NPU/SoC；
3. 在真实工程中先做 correctness baseline；
4. 若环境支持官方 `TopK` API，优先测试“Top-K logits → FP32 softplus/sqrt”；
5. baseline AC 后再根据真实 NPU/Judge 数据决定是否需要更底层优化。
