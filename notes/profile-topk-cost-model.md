# TopK 指令成本 profiling 协议

日期：2026-09-12

目标：停止继续凭算法复杂度猜 dav-2201 性能。对当前最佳 production 基线与失败的 pivot-filter 使用同一输入做 `msprof op simulator`，直接比较指令、流水线和代码行周期。

这一步只建立成本模型，不继续优化 pivot，也不把 simulator 时间当 Judge 时间。

## 当前状态

profiling 分支：`profile-topk-cost-model`。

首轮只测：

```text
rows = 1
E    = 1024
k    = 8
```

pivot-filter 在 CANNJudge 上出现高度集中的退化：case 3 `252.11 -> 849.19 us`，case 7 `59.09 -> 111.09 us`，case 9 `96.82 -> 178.44 us`，case 13 `634.57 us -> 1.26 ms`。因此现在的目标不是继续改算法，而是先回答 dav-2201 上各 primitive 实际贵在哪里。

当前 profiling harness 已改成 simulator/NPU 正式构建结构：device 端由 `ascendc_library` 构建，host 端单独编译并通过生成的 `<kernel>_do(...)` stub launch。这个新版结构尚未完成一次本机成功 profile，因此相关 issue #31 保持 open。

## 已踩坑

### 1. correctness probe 不能直接拿来 profiling

最初脚本调用 production correctness probe，并在命令后传 `1 1024 8`。但 pivot probe 的 `main()` 根本不读取命令行参数，而是固定跑完整 75-case，所以日志实际从 `E=64,k=2` 开始。

结论：profiling driver 必须只 launch 一次指定 shape，不能复用批量 correctness harness。

### 2. CPU twin-debug 的构建方式不能直接类比 simulator/NPU

单个 `.asc` 用 `add_executable` 的方式在 CPU twin-debug 下可以运行，但同样结构切到 simulator 后出现：

```text
[Bisheng] Launch kernel failure! ret 1
```

simulator 本身已正确加载 `dav_2201`，shape 也正确，因此问题在 kernel launch/build 链，而不是算法。

当前做法改为官方结构：

- device：`ascendc_library(... STATIC kernel_profile.asc)`；
- host：独立 `main.cpp`；
- host 链接生成的 sim kernel library；
- 通过 `<kernel>_do(...)` stub launch；
- profiling 构建启用 `-g -O2` 与 `ASCENDC_TRACE_ON`。

### 3. 编译 SoC 名与 simulator 名不是一套标识

本机 simulator 目录里存在 `Ascend920A`，`msprof op simulator --soc-version=Ascend920A` 也能正确加载：

```text
.../simulator/dav_2201/lib/config_stars.json
```

但 `ascendc_kernel_cmake` 的 `SOC_VERSION` 不接受 `Ascend920A`，支持列表使用 `ascend910b1`、`ascend910b2` 等编译 SoC 名。

当前默认映射：

```text
compile SOC_VERSION = ascend910b1
simulator model     = Ascend920A
```

脚本必须把这两个参数分开，不能把同一个字符串同时传给 CMake 和 msprof。

### 4. `msprof ... finish` 不等于 profiling 成功

child process 失败时，msprof 仍可能继续解析并打印：

```text
Op profiling finish. Welcome to next use.
```

但实际不会生成 `*_instr_exe.csv`。因此脚本现在把 CSV 存在性作为成功门槛；没有 instruction CSV 就立即退出，不运行另一边，也不生成 summary。

### 5. nested include / 增量构建仍需谨慎

本仓库已有 CMake 对嵌套 `.h/.inc` 依赖跟踪不稳定的历史。profiling harness 每次都在临时 worktree 中 clean configure/build，避免把 stale build 误认为新代码结果。

## 当前工具

- `tools/profile_topk_costs.sh`：建立 main / pivot detached worktree，生成 device-only 临时 kernel，分别采集同 shape profile；
- `tools/profile_harness/`：官方 simulator kernel-launch 结构；
- `tools/summarize_topk_profile.py`：聚合 instruction / pipe / code cycles；
- issue #31：记录 profiling 进度、失败原因与最终结论。

## 当前运行方式

```bash
git switch profile-topk-cost-model
git pull
source /home/leo/Ascend/cann-9.0.0/set_env.sh
bash tools/profile_topk_costs.sh Ascend920A 1 1024 8
```

脚本默认把 `Ascend920A` 映射到编译 `ascend910b1`。如本机后续确认需要别的具体 B1/B2/B3 编译型号，可以用脚本的显式 compile SoC 参数覆盖。

只有同时出现：

```text
[OK] baseline instruction CSV: ...
[OK] pivot instruction CSV: ...
```

才进入汇总：

```bash
python3 tools/summarize_topk_profile.py \
  profile-output/Ascend920A_r1_e1024_k8 \
  > profile-output/Ascend920A_r1_e1024_k8/summary.md
```

## 第一轮只回答这些问题

1. baseline 最贵的是 `Sort32`、`MrgSort`、数据搬运、同步还是其他工作；
2. pivot 新增周期主要来自 Compare、mask 读写、GatherMask、ReduceMax，还是 scalar/vector 依赖；
3. 两边主要 pipe 分布是否发生结构性变化；
4. Judge 中约 2-3x 的退化是否能由 simulator instruction cycle 同方向解释；
5. 基于真实占比，下一种算法即使完全消掉某一部分，理论最大收益是多少。

若首轮 profile 已经足以回答这些问题，不再批量跑更多 shape。
