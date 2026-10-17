# Deep Research 归档：SqrtSoftplusTopK 下一阶段优化方向

日期：2026-09-14

2026-10-18复核：实验当时尚不知道测试点形状。后来9月15日取得形状表，本次据此回看生产入口，dense/batch4对应覆盖为空，重复ReduceMax只命中C1。早期“模型到Judge失配”和路线无效的归因需要修正，具体见[完整复盘](contest-retrospective-2026-10-18.md)。

## 1. 当前问题与实验约束

目标算子：`SqrtSoftplusTopK`。输入 `router_logits(T,E)`，`E=64..1024`，`k=1..64`，FP16/FP32，目标平台为 dav-2201 / Atlas A2/A3 类 Ascend C 环境。

由于 `sqrt(softplus(x))` 单调递增，可以先在 raw logits 上做 exact Top-K，再只对选中值做变换。最终输出只需要 scores，不需要 indices。

当前 dense baseline Judge（15 点）约为：

`2.37, 3.89, 252.11, 8.09, 17.07, 24.12, 59.09, 87.29, 96.82, 244.48, 252.67, 650.03, 634.57 us, 2.13 ms, 3.46 ms`。

最新榜首参考约为：

`1.74, 2.25, 40.66, 2.76, 3.66, 7.04, 9.00, 13.62, 12.73, 27.36, 37.22, 30.80, 59.27, 40.81, 418.73 us`。

差距在部分 case 达到约 6x、20x、50x，因此下一阶段重点应放在改变 selection 主结构，而不是继续做百分之几到十几的局部微调。

### 已经排除或明显降级的方向

1. **pivot / threshold + mask candidate compaction**
   - exact pivot-filter 已做正确实现。
   - 真实 Judge 总体约退化 17.5%，若干 case 大幅退化。
   - 原因与 profile 一致：`Compare -> GetCmpMask -> GetValue -> scalar count/branch -> GatherMask` 形成频繁 vector→scalar 串行依赖。
   - 后续若再用 threshold，只能作为固定控制流中的数据屏蔽，不能再按 mask cardinality 动态控制路径。

2. **继续优化 Sort32 / MrgSort 本身**
   - dav-2201 simulator 中 `VBS32` / `VMS4V2` 并非主要 instruction-cycle 热点。
   - current prefix/dense 路径的数学 primitive 本身不是已观察到的最大账面开销。

3. **bulk prefix pack / barrier 微调**
   - 将逐 queue pack 改为单次 strided DataCopy 后，simulator 中 BAR cycles 约 -76%，VECTOR cycles 约 -71%。
   - 真实 Judge 基本 0 收益。
   - 结论：simulator 的 pipe/instruction cycle 适合机制诊断，但不能直接预测真实 NPU wall time。

4. **E=1024,k=8 的跨行 batching**
   - rows scaling：`rows=1/2/4/8 -> 2.89/4.52/7.80/14.34 us`，约可拟合 `1.25 + 1.64*rows us`。
   - 固定形状 specialization 只有约 2.4% simulator 收益。
   - batch4 本身在 simulator 中约 17.3% 独立收益，但 production Judge 基本完全持平。
   - 结论：减少 Ascend C API 发射、scalar store / flag / wait 的 simulator 成本不足以构成真实性能主线。

5. **编译期专门化**
   - `bulk_control 6.17 us -> specialized_sequential 6.02 us`，仅约 2.4%。
   - 不作为主方向。

## 2. CANN / Ascend 官方资料给出的关键事实

### 2.1 A2/A3 上官方 TopK 仍以 merge-sort 路径为主

CANN 9 `TopK` 文档说明 TopK 有 `MERGE_SORT` 与 `RADIX_SELECT` 两种算法，但 `RADIX_SELECT` 只支持 Atlas 350 Accelerator Card。Atlas A2/A3 上没有通过 `TopKConfig` 直接切到官方 radix-select 的捷径。

官方文档还明确描述：

- TopK small mode 使用 `Sort32`，随后 `GatherMask`；
- normal mode 内部也是 merge-sort 类结构；
- 当 `isInitIndex=false` 时，API 仍会内部生成 index；
- A2/A3 上 `isReuseSrc` 等参数为保留参数，并不是可利用的性能开关。

因此：

- “换一下 TopK tiling/config 就得到完全不同算法”的可能性显著下降；
- 当前手写 `Sort32 -> prefix -> MrgSort` 与官方通用 TopK 主结构实际上接近；
- 若想获得数量级变化，更可能需要绕开“整行 pair 排序/归并”本身，而不是继续模仿 merge-sort。

官方资料：

- CANN 9 TopK API：<https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0836.html>

### 2.2 ReduceMax 是一个有价值的小-k替代 primitive

CANN 提供 ReduceMax / 带 index 的最大值归约能力。对于 `k` 极小的场景，可以考虑 repeated argmax：每次扫描得到最大值及位置，将该位置置为 `-inf` 后继续下一轮。

这个结构的关键吸引力不是“大 O 更好”，而是：

- 不需要给整行元素一路携带 index pair 做完整排序归并；
- 最终只输出 scores，本题不需要保存完整 indices；
- 与 pivot-filter 不同，不需要对每 64 个元素提取 mask、计数候选、动态分支和 compaction；
- 每一轮只产生一个 argmax 结果和一次单点失效。

风险也明确：若 k 增大，重复扫描会线性增加，因此只适合作为小-k specialization，而不是统一算法。

### 2.3 华为公开 MoE gating Top-K 比泛化 GPU Top-K 更值得参考

Huawei `hicann/ops-transformer` 已公开 `aclnnMoeGatingTopK/V2`，支持 Atlas A2/A3，并覆盖 MoE routing 场景；专家数可到 2048，语义与本题比一般排序 benchmark 更接近。

公开文档显示 MoeGatingTopKV2 的主流程包括：

- 对专家分数做 norm；
- 可选 group selection；
- Top-K 专家选择；
- gather 选中专家得分；
- 输出 score 与 expert index。

尤其 group selection 里会先用 `ReduceMax` 或 group 内 `TopK(k=2)` 得分，再把未选中组置为 `-inf`，说明“先缩小 active 范围、再做 TopK”是华为 routing 算子真实采用的结构思想之一。

资料：

- ops-transformer：<https://github.com/hicann/ops-transformer>
- MoeGatingTopKV2：<https://github.com/hicann/ops-transformer/blob/master/moe/moe_gating_top_k/docs/aclnnMoeGatingTopKV2.md>

下一阶段源码调查重点不是接口文档，而是 A2/A3 kernel：expert dimension 如何切分，小 k 是否走独立路径，index payload 如何管理，每 token/row 如何映射到 core。

## 3. HPC / GPU Top-K 相关方向的适用性判断

### 3.1 Radix selection

RadiK（ICS 2024）证明 GPU 上 radix Top-K selection 可以显著减少全排序成本，并针对多 GPU/大规模数据做了专门优化。

论文：

- RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection, ICS 2024
- DOI: <https://doi.org/10.1145/3650200.3656596>

但在 dav-2201 上直接照搬 radix/threshold scan 有两个问题：

1. 官方 RADIX_SELECT 不支持 A2/A3；
2. 如果每一轮 radix/threshold 都要 `Compare -> mask -> scalar count -> branch`，会重新落入 pivot-filter 已经实测失败的 vector↔scalar 串行模式。

因此 radix 目前只作为算法思想参考，不排在第一 production spike。

### 3.2 bitonic / full sorting network

对 E<=1024 的短向量，GPU 上 bitonic / register sorting network 常用于 warp/block 级 selection。但 Ascend C 当前公开 primitive 已经提供 Sort32/MrgSort；我们自己的 profile 也没有显示 Sort32/MrgSort primitive 本身是主要瓶颈。

在没有 register/shuffle 风格低层指令暴露的情况下，重新手写 sorting network 预期收益不高，优先级较低。

### 3.3 register-level / warp-level selection

GPU 高性能 Top-K 常把每线程候选留在 register，再通过 warp shuffle / ballot / shared merge 完成 selection。这类方案的核心优势来自廉价的 thread-lane communication 与 register-resident candidate set。

当前公开 Ascend C 抽象并没有提供一套与 CUDA warp shuffle/ballot 完全对等、可直接移植的低成本接口；之前 pivot 的 mask→scalar 往返也说明不能机械搬 GPU warp-level 算法。

若后续在华为公开 MoE kernel 中发现更底层 intrinsic 或固定宽度 vector register selection，则应重新提升优先级。

## 4. 下一阶段主方向排序

### 方向 1：repeated ReduceMax / argmax，小 k production spike

优先级：**最高，立即 Judge 验证**。

适用：FP32，小 k（第一版 `k<=4`），E=64..1024。

最小结构：

```text
load row to UB
repeat k times:
    ReduceMax / argmax over row
    save max value
    set selected lane to -inf
transform selected values
write scores
```

为什么值得做：

- 完全绕过整行 `(value,index)` 排序归并；
- 真正利用“只要 scores、k 很小”的题目结构；
- 与失败 pivot 不同，不需要 mask cardinality / candidate compaction；
- 可以很快通过真实 Judge 证伪。

停止条件：

- 受影响 case 只出现噪声级或十几个百分点改善，则停止；
- 若中大型受影响 case 出现约 1.5x~2x 或更大改善，再扩 shape/k。

### 方向 2：fixed-control rolling segmented exact Top-K

优先级：**第二**。

适用：k 超出 repeated ReduceMax 甜点区，尤其 E=512/1024。

目标结构：

```text
fixed-size segment scan
maintain fixed-width current top-k
use kth threshold only as data mask
new segment values below threshold -> -inf
merge current candidates + fixed segment with exact small selection
repeat
```

关键约束：

- 不根据 candidate count 走动态路径；
- 不做 `GetCmpMask -> GetValue -> scalar count -> compact`；
- threshold 只决定 lane 是否被写成 `-inf`，控制流固定。

该结构直接攻击当前 prefix merge 的根本问题：所有 32-element chunk 都先变成携带 index 的排序 prefix，再一路 merge；rolling candidate set 尝试让每轮只保留与最终 k 有关的固定宽度候选。

### 方向 3：深入对照华为 MoeGatingTopK A2/A3 源码

优先级：**与方向 2 并行做资料分析，但不先写通用框架**。

重点回答：

- 是否按 expert count/k 做多个 kernel specialization；
- 是否存在 repeated argmax / group reduction / rolling candidate；
- 是否用不同于通用 TopK API 的专门低层 primitive；
- token/row 与 vector core 的映射方式；
- index 是全程携带还是只在末端生成；
- 是否存在 `TopKV2_high_performance` 一类专用实现路径。

如果源码显示 Huawei 自己在 A2/A3 routing Top-K 中使用了我们尚未尝试的低层结构，应优先跟随真实平台实现，而不是继续依据 GPU 类比。

## 5. 降级方向

短期内不继续投入：

- prefix packing / barrier 微调；
- B=8 / bank conflict / final merge batching；
- 单纯减少 Sort32/MrgSort 次数；
- 固定 shape compile-time specialization；
- 继续调 pivot/filter 的迭代次数和 threshold；
- 自己手写 dav-2201 radix selection，除非先找到避免 vector↔scalar count/branch 的固定控制实现；
- 仅靠 simulator percentage 判断是否值得上 Judge。

## 6. 对隐藏 Judge shape 的认识

目前无法仅从 15 个时间点可靠反推出全部 `(rows,E,k,dtype)`。pivot 只影响部分 case，说明不同 case 确实落在不同 dispatch 路径；但没有 shape 元数据时，继续用时间数字强行猜 E/k 风险很高。

因此后续 production spike 的设计应尽量使用清晰 gate，使“哪些 case 被改动”可由代码本身确定；再通过 Judge 的逐点变化做 shape 指纹推断。

## 7. 当前决策

下一步不再做新的 simulator 性能准入实验。

先做一个最小 `repeated ReduceMax` production spike，只覆盖 `FP32 && k<=4`，其他 shape 完全回退当前干净 baseline。完成最小 correctness 后直接跑完整 Judge。

如果这个方向不能在真实 NPU 上产生明显倍数级收益，则停止并转入 rolling segmented / Huawei MoeGatingTopK 实现对照。

对应 issue：

- #40 `production spike：k<=4 repeated ReduceMax exact TopK`
- #41 `后续方向：rolling segmented exact TopK 与华为 MoE gating 实现对照`
