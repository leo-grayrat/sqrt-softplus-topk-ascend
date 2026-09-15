# Vendor Perf TopK 第七轮：transform 内部 PIPE_V 自动同步 A/B

日期：2026-09-15

## 出发点

第五轮和第六轮已经把后处理成本拆得比较清楚：

- 把逐行 `TransformSelected(8)` 合并成一次跨行 batched transform，case 7/9/13 出现约 38% 的真实 Judge 收益；
- 在已经只调用一次 transform 的前提下，再把元素数从 `validRows*8` 减到 `validRows*k`，第六轮总体几乎持平。

因此当前更值得检查的是 `StableSoftplusSqrt` 这一次调用内部的固定指令/同步成本，而不是继续压缩元素数。

当前 FP32 transform 为：

```text
Abs -> barrier
Muls -> barrier
Exp -> barrier
Adds -> barrier
Ln -> barrier
Maxs -> barrier
Add -> barrier
Sqrt -> barrier
```

## CANN/BiSheng 依据

CANN 9.0 BiSheng 文档说明：

- `--cce-auto-sync` 用于根据 LocalTensor 数据依赖自动插入核内同步；
- kernel 直调和自定义算子工程默认开启自动同步；
- `PIPE_V` 单流水同步可由编译器自动插入；
- 正确插入依赖所有相关函数为 inline，并依赖规范的 LocalTensor 直接逻辑关系。

本仓库 kernel 函数均为 `__aicore__ inline`，vendor transform 内部直接对同一 `LocalTensor` 变量进行读写，属于适合做该 A/B 的范围。

参考：

- CANN 9.0 BiSheng 编译选项：`--cce-auto-sync` / `--cce-auto-sync-log`
- Ascend C 同步控制：默认自动同步时，PIPE_V 由编译器插入

## 第七轮 bounded change

基线：`vendor-perf-topk-round6-extract-k`。

只针对 FP32、`2<=k<=8` 的 vendor-perf 路径新增：

```cpp
VendorStableSoftplusSqrtAutoSync(...)
```

它保持：

- 完全相同的稳定 softplus 数学表达式；
- 完全相同的八个 vector op；
- 完全相同的运算顺序；
- GatherMask/selection/merge/staging/true-k extraction/copyout 全部不变。

唯一变化：

- 删除 `Abs/Muls/Exp/Adds/Ln/Maxs/Add/Sqrt` 之间七个手写 `PipeBarrier<PIPE_V>()`；
- 依赖 BiSheng 自动同步按真实 LocalTensor 依赖重新插入同步；
- `Sqrt` 后仍保留一个显式 `PipeBarrier<PIPE_V>()`，不改变 transform 到 output queue 的边界行为；
- `ExtractKValues()` 末尾原有 barrier 也保留，不改变 GatherMask 到 transform 的边界行为。

因此该轮只回答一个问题：手写 barrier 是否让这一小段 transform 比编译器依赖分析生成的同步更保守。

## 预期解释

如果 Judge 明显变快：说明当前手写同步过度串行化，尤其 `Ln(tmp)` 与 `Maxs(values)` 这类不直接依赖的相邻 vector op 可能被无必要地完全 commit-serialize。

如果基本持平：说明 BiSheng 最终生成的同步与手写版本等价，或 barrier 本身并不是第五轮观察到的固定调用成本主因；下一步应回到 selection/merge 或算子数量本身。

如果 correctness 出问题：首先检查实际比赛编译流程是否确实启用了 `--cce-auto-sync`，不直接把问题归因于数学实现。

## 验证

1. `python3 tools/check_vendor_perf_topk.py`；
2. Bisheng/CANN production CPU probe；
3. 用 `-g --cce-auto-sync --cce-auto-sync-log=<file>` 编译一次 device 目标，确认编译器实际插入 vector 同步；
4. 根据 auto-sync 日志决定是否值得进入 Judge。

### auto-sync 日志参数注意事项

实际编译确认：`--cce-auto-sync-log=<file>` 只能在 debug 编译下使用，必须同时传入 `-g`。若只传 `--cce-auto-sync-log`，BiSheng 会在参数检查阶段报：

```text
invalid argument '-cce-auto-sync-log' only allowed with '-g'
```

日志专用 device build 应使用：

```bash
rm -rf /tmp/topk-round7-device /tmp/topk-round7-auto-sync.log

cmake \
  -S submission \
  -B /tmp/topk-round7-device \
  -DNPU_ARCH=dav-2201 \
  -DCMAKE_ASC_FLAGS="-g --cce-auto-sync --cce-auto-sync-log=/tmp/topk-round7-auto-sync.log"

cmake --build /tmp/topk-round7-device -j1
```

该参数修正只影响调试日志生成，不改变第七轮 kernel 实现。

## 实际结果

结构 checker 通过，production CPU probe 共 224 个 case 全部 PASS，覆盖 `k={2,3,4,5,7,8}`、多种对齐/非对齐 E、尾 batch 与多核场景。

auto-sync 日志则直接给出了否定结论。`VendorStableSoftplusSqrtAutoSync` 中 54--61 行依次对应：

```text
Abs
Muls
Exp
Adds
Ln
Maxs
Add
Sqrt
```

BiSheng 在这八行前全部记录了自动插入：

```text
PipeBarrier<PIPE_V>()
```

因此删除手写中间 barrier 后，编译器没有利用 `Ln(tmp)` 与 `Maxs(values)` 等表面上的独立数据流来形成明显重叠，而是采取了保守的逐向量算子同步。再加上函数末尾保留的显式 barrier，这个版本最终的同步结构与第六轮基本等价；`Abs` 前还存在额外的自动同步记录。

### 结论

第七轮不进入真实 Judge。

原因不是 correctness 风险，而是编译器日志已经证明该 A/B 在最终生成同步结构上没有实质性变化。继续 Judge 只能重复验证一个已经由 codegen 诊断回答的问题。

这进一步说明第五轮 batched transform 的大收益不能解释成“去掉了 transform 内部 barrier”；更可信的解释仍是减少了短向量 API 序列的重复发射次数。下一步停止 transform barrier 微调，回到 Huawei vendor Perf selection/merge 执行组织，检查当前实现是否还存在逐 chunk/逐段调用，而 vendor 原版能够通过 repeat/block 参数一次批量完成的地方。
