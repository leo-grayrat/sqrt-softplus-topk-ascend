# SqrtSoftplusTopK 仓库设计

## 目标

这个仓库只做三件事：保存官方题意与数学参考、接入 CANNJudge 的真实空工程、记录真实验证结果。不要提前搭一套自己想象的 CANN 工程。

## 第一版结构

```text
README.md
AGENTS.md
reference/
    reference.py
tests/
    test_reference.py
submission/
    README.md
notes.md
```

`submission/` 只接收从 CANNJudge 下载的真实题目工程。在拿到它之前，不自行创建 `op_host/`、`op_kernel/` 等目录。

## 正确性

官网 PyTorch 参考实现是唯一正确性基准。另写一份“先 Top-K logits，再计算 `sqrt(softplus)`”的等价实现，用测试与官方参考逐项比较。

这个变换成立，因为 `sqrt(softplus(x))` 单调递增。题目只输出分数，不输出专家 ID，所以并列值不会带来索引顺序问题。

测试至少覆盖：FP16/FP32、k=1/6/64、E=64/非 2 的幂/1024、20 附近、极大正负值、`-inf`、重复值和随机输入。

## 第一版 Ascend C 路线

拿到真实 starter project 并确认比赛环境后，优先尝试官方 `TopK` API：先对 logits 做 Top-K，再只对 k 个结果做 FP32 softplus 与 sqrt。若比赛环境支持 `TOPK_NORMAL` 与 `TOPK_NSMALL`，两者都实测。

只有官方 TopK 的真实性能不够好时，才考虑 Sort32、自定义 small-k、专门的 k=1 reduction 等更底层方案。

## 工作原则

1. 不猜 CANN/Ascend C API，使用与比赛环境匹配的官方资料或 starter project。
2. 不自己造比赛接口或模拟环境来证明实现有效。
3. 本地测试通过、编译通过、Judge AC、性能提高是四件不同的事。
4. 性能结论必须来自真实 NPU/Judge 数据；失败方案也记到 `notes.md`，避免重复试。

## 当前未知

- 比赛实际 CANN 版本
- 实际 NPU/SoC
- CANNJudge starter project 的具体结构

这些信息拿到真实工程后再补，不提前猜。
