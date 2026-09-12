# TopK 指令成本 profiling 协议

日期：2026-09-12

目标：停止继续凭算法复杂度猜 dav-2201 性能。对当前最佳 production 基线与失败的 pivot-filter 使用同一输入做 `msprof op simulator`，直接比较指令、流水线和代码行周期。

## 为什么做这一步

pivot-filter 在 CANNJudge 上出现高度集中的真实退化：case 3 `252.11 -> 849.19 us`，case 7 `59.09 -> 111.09 us`，case 9 `96.82 -> 178.44 us`，case 13 `634.57 us -> 1.26 ms`；其他多条 fallback/非目标路径基本不变。说明“减少排序工作量”不能直接推出 Ascend C primitive 映射更便宜。

这一步只建立成本模型，不继续优化 pivot，也不把 simulator 时间当 Judge 时间。

## 官方工具依据

CANN 的算子调优工具支持：

- `msprof op`：上板采集，输出 `PipeUtilization.csv`、`ArithmeticUtilization.csv`、`MemoryUB.csv`、`ResourceConflictRatio.csv` 等；
- `msprof op simulator`：仿真采集，输出每核 `*_instr_exe.csv`、`*_code_exe.csv`、`trace.json`；
- `*_instr_exe.csv` 直接包含 `instr / pipe / call_count / cycles / running_time(us)`；
- `*_code_exe.csv` 直接包含 `code / call_count / cycles / running_time(us)`。

仿真时 `--soc-version` 要给具体型号，例如 `Ascend910B1`，不能只写 `Ascend910B`。

## 第一次采集形状

先固定：

```text
rows = 1
E    = 1024
k    = 8
```

理由：这一步分析单行 selector 的内部成本，不测端到端吞吐。单行可避免大量核心和多行循环把流水图弄得难读，同时 E=1024 能完整暴露 prefix merge 与多轮 pivot 扫描。

后续只有在这份 profile 无法解释 Judge 退化时，才考虑第二个形状；不要预先批量跑很多 shape。

## 操作

先切到 profiling 分支并配置 CANN：

```bash
git fetch origin
git switch profile-topk-cost-model
git pull
source /home/leo/Ascend/cann-9.0.0/set_env.sh
```

确认本机安装了哪些 simulator 型号：

```bash
find /home/leo/Ascend/cann-9.0.0/tools/simulator \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null || true
```

如果该目录结构不同，可用：

```bash
find /home/leo/Ascend/cann-9.0.0 -type d -path '*/simulator/Ascend*' -printf '%f\n' 2>/dev/null | sort -u
```

选实际存在的具体型号，例如 `Ascend910B1`，然后：

```bash
bash tools/profile_topk_costs.sh Ascend910B1 1 1024 8
```

脚本会：

1. fetch `origin/main` 与 `origin/pivot-topk-rewrite`；
2. 在 `/tmp` 创建两个 detached worktree；
3. 两边都用 `CMAKE_ASC_RUN_MODE=sim`、`dav-2201` clean build；
4. baseline 执行 `topk_prefix_production_probe 1 1024 8`；
5. pivot 执行 `topk_pivot_production_probe 1 1024 8`；
6. 将 msprof 原始数据写入 `profile-output/<soc>_r1_e1024_k8/{baseline,pivot}`。

汇总：

```bash
python3 tools/summarize_topk_profile.py \
  profile-output/Ascend910B1_r1_e1024_k8 \
  > profile-output/Ascend910B1_r1_e1024_k8/summary.md
```

## 第一轮分析只回答这些问题

1. baseline 最贵的指令到底是 `Sort32`、`MrgSort`、数据搬运还是其他同步/标量工作；
2. pivot 新增周期主要来自 Compare、mask 读写、GatherMask、ReduceMax，还是 scalar/vector 依赖空洞；
3. baseline 与 pivot 的主要 pipe 分布是否发生结构性变化；
4. Judge 中约 2-3x 的退化能否在 simulator 的 instruction cycle 差值中得到同方向解释；
5. 基于真实占比，下一种算法如果彻底消掉某一部分，理论最大收益是多少。

如果这些问题已经得到清楚答案，不再继续 profiling 更多 shape；直接据此选择下一条算法或决定停止。
