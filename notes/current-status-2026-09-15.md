# 当前状态（2026-09-15）

## Canonical baseline

`main` 当前以 `vendor-perf-topk-round6-extract-k` 的已验证实现为基线，对应历史提交 `ecf13245646eaa9ce03cb8245638f7ec7366e7a4`。

选择 Round 6 作为主线的原因不是它在每个测试点都绝对最快，而是它同时满足：

- 本地结构 checker PASS；
- CANN 9.0 / Bisheng 编译通过；
- production CPU probe PASS；
- 真实 CANNJudge 15/15 Pass；
- 相比 Round 5 结构更接近 Huawei vendor perf 的 true-k value extraction，且没有引入已知的大幅性能退化。

Round 6 Judge：

```text
1   2.33 us
2   3.77 us
3   139.80 us
4   7.80 us
5   16.63 us
6   23.95 us
7   17.73 us
8   86.99 us
9   27.71 us
10  244.10 us
11  252.30 us
12  650.44 us
13  176.97 us
14  2.13 ms
15  3.46 ms
```

## Round 5 / 6 已确认的关键机理

Round 5 把逐行 `TransformSelected` 改成跨行 batch 后，case 7/9/13 分别出现约 38%~40% 的时间下降。随后 A/B 证明：

- 把 transform 恢复成逐行调用，主要收益基本消失；
- 把 batch copyout 恢复成逐行，性能几乎不变；
- Round 6 再把单次 transform 的元素数从固定 `8*rows` 缩到真实 `k*rows`，只得到小幅或噪声级变化。

因此当前最可靠的解释是：主要收益来自**减少很多很短的 transcendental vector 调用及其固定发射/同步开销**，而不是简单减少几个参与变换的元素。

## Round 7~10 的位置

这些分支不再视为主线，而作为 Round 6 之后的实验记录。

### Round 7：auto-sync transform

`vendor-perf-topk-round7-auto-sync-transform`

删除 transform 内部手写 `PIPE_V` barrier 后，Bisheng auto-sync 日志显示它在 `Abs / Muls / Exp / Adds / Ln / Maxs / Add / Sqrt` 前重新插入了同步。production probe 224/224 PASS，但机制上没有形成新的执行重叠，因此没有进入 Judge。

结论：负结果，停止该方向。

### Round 8：larger batch

`vendor-perf-topk-round8-larger-batch`

把旧 `16 / 8 / 4` 行容量改成约 `5120 / inner`。Judge 15/15 Pass：

```text
case 3   132.97 us
case 7    20.65 us
case 9    27.43 us
case 13  172.43 us
```

相对 Round 6：case 3/9/13 有收益，但 case 7 明显退化，15 点总量没有形成净收益，因此不作为 main。

### Round 9：cap32

`vendor-perf-topk-round9-cap32`

在 Round 8 基础上增加 32-row 上限。Judge 结果与 Round 8 几乎相同：

```text
case 3   132.89 us
case 7    20.50 us
case 9    27.30 us
case 13  172.59 us
```

现在有了测试点 shape 后可以确认：对 3/7/9/13，Round 9 的 32-row cap 实际上没有改变它们的 batchRows，因此这次实验没有真正切到目标条件。

### Round 10：restore small tier

`vendor-perf-topk-round10-restore-small-tier`

该分支把 `inner<=256` 恢复为 16 rows，但尚未 Judge。高可信测试点 shape 信息出现后，这个实验的解释力已经降低：它会同时改变 case 7、9、13，而三者对 batch 的反应并不相同。因此暂不把 Round 10 视为下一生产候选。

## 高可信测试点 shape 信息

以下数据来自参赛者泄露并经过交叉验证，可信度很高，但并非赛事官方公开资料，因此仓库中按“高可信外部信息”记录，不当作正式 contract。

| case | T | E | k | dtype |
| ---: | ---: | ---: | ---: | --- |
| 1 | 8 | 64 | 4 | fp32 |
| 2 | 16 | 97 | 5 | fp16 |
| 3 | 7777 | 888 | 8 | fp32 |
| 4 | 64 | 256 | 4 | fp16 |
| 5 | 512 | 200 | 16 | fp16 |
| 6 | 1024 | 130 | 15 | fp32 |
| 7 | 3072 | 160 | 7 | fp32 |
| 8 | 4096 | 256 | 64 | fp16 |
| 9 | 5000 | 150 | 6 | fp32 |
| 10 | 16384 | 128 | 16 | fp16 |
| 11 | 16384 | 97 | 32 | fp32 |
| 12 | 32768 | 256 | 3 | fp16 |
| 13 | 32999 | 210 | 8 | fp32 |
| 14 | 65536 | 512 | 10 | fp16 |
| 15 | 65555 | 1024 | 64 | fp32 |

这张表与历史实验响应高度吻合。例如早期 vendor-perf 仅允许 `E%32==0` 时，case 7 明显提速而 3/9/13 基本不动；放宽非 32 对齐 gate 后，3/9/13 同时出现结构性下降。

## 当前路径覆盖

Round 6 vendor-perf 主路径只覆盖 FP32 `2<=k<=8`，因此对应公开 shape 后的 case 为：

```text
1, 3, 7, 9, 13
```

其中 case 1 只有 8 行，启动成本占主导；真正决定该路径收益的主要是 3/7/9/13。

其余 10 个点仍主要依赖旧路径，可按以下三组看：

- FP16 small-k：2 / 4 / 12；
- FP32 medium-large-k：6 / 11 / 15；
- FP16 medium-large-k：5 / 8 / 10 / 14。

这意味着当前最大的结构性空间已经不在继续微调 FP32 small-k 的几个百分点，而在**把已证明有效的执行组织扩到新的 dtype / k 区间**。

## 分支与 issue 约定

从本次整理开始：

- `main` = 当前可信生产基线；
- 实验分支只服务于一个明确假设；
- 成功结果整理后回 `main`，失败结果归档到 notes，不再让 `roundN` 分支持续承担主线职责；
- 现有历史分支暂时保留以避免破坏追溯，后续无需继续从它们分叉开发；
- #43 在本次整理中作为 vendor-perf FP32 small-k 阶段的收尾 issue，新的优化阶段另行规划。
