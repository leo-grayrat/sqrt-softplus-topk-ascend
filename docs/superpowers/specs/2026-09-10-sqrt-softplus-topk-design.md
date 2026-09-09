# SqrtSoftplusTopK 竞赛仓库设计

## 目标

建立一个面向 2026 CANN 挑战赛初赛 SqrtSoftplusTopK 算子的最小工程仓库，用于：

1. 固定赛题规则、数学化简和正确性参考；
2. 为 Ascend C 实现提供明确入口，但不在没有官方资料或真实环境验证时臆造 API；
3. 记录每次编译、正确性、Judge 与性能实验，避免把“看起来合理”误当成“已经验证”；
4. 支持后续由 AI 辅助进行实现、调试和性能迭代。

## 核心数学观察

赛题要求输出每行 `sqrt(softplus(x))` 的 Top-k 值。因为 `softplus(x)` 与 `sqrt(x)` 在定义域上均严格单调递增，因此：

```text
TopK(sqrt(softplus(x))) == sqrt(softplus(TopK(x)))
```

所以候选实现的首要优化方向是：先在原始 logits 上完成 Top-k，只对选中的 `T * k` 个元素进行 FP32 softplus 与 sqrt，再按输入 dtype 输出。

这一等价变换必须由 reference/test 覆盖；在真实 Judge 通过之前，只称为“候选优化”，不宣称最终实现已正确或更快。

## 仓库结构

```text
sqrt-softplus-topk-ascend/
├─ README.md
├─ AGENTS.md
├─ .gitignore
├─ docs/
│  ├─ problem.md
│  ├─ key-observations.md
│  ├─ optimization-plan.md
│  └─ environment.md
├─ reference/
│  └─ reference.py
├─ operator/
│  └─ README.md
├─ tests/
│  ├─ README.md
│  └─ cases.py
├─ benchmark/
│  ├─ README.md
│  └─ results.csv
└─ notes/
   └─ experiments.md
```

## 各模块职责

### `reference/`

只负责 PyTorch/CPU 侧数学参考实现，包括：

- 官方定义版本：先全量 `sqrt(softplus)`，再 sort/top-k；
- 数学化简版本：先对 logits top-k，再做 transform；
- 二者的一致性检查。

reference 不承载 Ascend C 性能逻辑。

### `tests/`

生成和组织边界与随机 case，至少覆盖：

- FP16 / FP32；
- `k=1`、常见 `k=6`、`k=64`；
- `E=64`、非 2 的幂、`E=1024`；
- 极大正数、极大负数、`-inf`、重复值；
- 小 T 与较大 T。

本地 reference 测试只能证明数学/host 侧逻辑；不能替代 CANN 编译、NPU 运行或官方 Judge。

### `operator/`

Ascend C 实现目录。第一版仅建立入口说明，不编造 kernel 文件结构。实际文件必须依据当前比赛模板、官方 sample 或已经确认可用的 CANN API 创建。

### `benchmark/`

只保存真实测量结果。每条性能记录需要注明环境、版本、case、实现版本和测量来源。

### `notes/experiments.md`

按时间记录实验：假设、改动、验证方式、结果、结论。失败方案也保留，避免 AI 在后续会话中反复重试已失败路径。

## 验证层级

仓库统一使用以下状态，不得混用：

1. **静态构思**：只有代码或方案，未运行；
2. **Host/reference 通过**：Python 等参考测试通过；
3. **编译通过**：在明确的 CANN 环境中成功编译；
4. **NPU 运行通过**：在真实昇腾硬件上运行且输出正确；
5. **Judge AC**：官方隐藏 case 全部通过；
6. **性能改善已验证**：同一可比环境下有真实测量证明更快。

任何较低层级不得被描述成较高层级。

## AI 协作原则

- 不根据记忆猜 CANN/Ascend C API 名称、参数或可用版本；不确定时查官方文档/sample 或明确标注未知。
- 不自己设计“模拟 CANN 接口”来证明实现有效。
- 不因本地测试全绿就宣称比赛实现可提交。
- 性能优化必须对应真实 benchmark/Judge 数据；没有数据时只能提出假设。
- 优先保留简单、可验证的 correctness baseline，再逐步加入 specialization。
- 对 `k=1`、small-k、large-k、不同 E/dtype 的专门路径只在数据支持后加入，避免提前堆复杂分支。

## 第一阶段交付

第一阶段只完成工程骨架与 host 侧 reference/test：

1. README 与赛题摘要；
2. AGENTS.md 约束；
3. 官方数学 reference；
4. 单调性化简 reference；
5. case 生成与 host 侧一致性测试；
6. Ascend C/operator 目录只放接入说明；
7. benchmark/experiments 模板。

明确不在第一阶段声称：Ascend C 已实现、可编译、可在 NPU 运行、已 AC 或已获得性能提升。

## 后续方向

获得真实比赛模板或官方 sample 后，再进入 Ascend C baseline：

1. 复用已验证的工程结构与官方 API；
2. 先追求 15 case 正确；
3. 再根据 Judge/benchmark 结果研究 raw-logits Top-k、small-k specialization、tile/block 设计与数据搬运；
4. 每次只做可归因的性能改动并记录结果。
