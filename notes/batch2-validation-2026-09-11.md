# FP32 hierarchical batchRows=2 CPU 验证

日期：2026-09-11

在 `debug/cpu_topk_probe` 增加双行模式后，用户本地执行：

```bash
./build/topk_hierarchical_probe --batch2
./build/topk_hierarchical_probe --full-batch2
```

两组均全部通过。

含义：

- FP32 hierarchical 的两行输入布局、候选布局、final NORMAL `outter=2`、输出 stride 与 host golden 在 CPU twin debug 下均通过；
- 之前线上 `batchRows=2` 不是 Runtime Error 根因的判断得到进一步支持；
- 该结果只属于 CPU correctness / API-contract 层，不代表 NPU 性能。

随后正式 `submission/kernel.asc` 仅恢复 FP32 hierarchical 的 1/2 行 batch 选择，FP16 继续走旧 NORMAL 路径，不同时加入 shape gate 或其他优化。对应提交：`363b922`。