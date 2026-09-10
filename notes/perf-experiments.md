# 性能实验记录

## Baseline：逐行 TopK

日期：2026-09-10

状态：CANNJudge Pass，得分 22.80，排名 63。

实现：每个 aicore 按行循环；每行补齐 E 后调用一次 `TopK(..., TOPK_NORMAL)`，再对该行的 k 个结果做 FP32 softplus + sqrt。

15 个 case（我方 / TBest）：

```text
1   2.74us   / 1.82us
2   3.55us   / 2.38us
3   407.01us / 41.52us
4   7.36us   / 2.80us
5   16.95us  / 5.34us
6   27.19us  / 7.04us
7   67.44us  / 9.64us
8   101.10us / 28.52us
9   110.70us / 14.18us
10  291.63us / 34.91us
11  314.56us / 43.80us
12  779.59us / 34.08us
13  743.75us / 61.64us
14  2.45ms   / 234.39us
15  3.92ms   / 537.28us
```

结论：正确性已确认；性能差距随 case 变化很大，不像单纯固定启动开销。对应 issue #7。

## 下一实验：双行 TopK

只验证一个变量：当 T 足够大、不会损失核心并行度时，每个 aicore 一次把两行组成 `outter=2` 调用 TopK。小 T 继续保持 `outter=1`。其余数学计算和 padding 逻辑不改。

结果：待本地编译和 CANNJudge 实测。
