# TopK 指令成本 profiling 协议

日期：2026-09-12

目标：停止继续凭算法复杂度猜 dav-2201 性能。对当前最佳 production baseline 与失败的 pivot-filter 使用同一输入做 simulator profiling，直接比较指令、流水线和代码行周期。

首轮只测：

```text
rows = 1
E    = 1024
k    = 8
```

pivot-filter 在 CANNJudge 上出现高度集中的退化：case 3 `252.11 -> 849.19 us`，case 7 `59.09 -> 111.09 us`，case 9 `96.82 -> 178.44 us`，case 13 `634.57 us -> 1.26 ms`。当前目标不是继续优化 pivot，而是先回答 dav-2201 上各 primitive 实际贵在哪里。

## 当前方案

profiling 分支：`profile-topk-cost-model`。

当前 harness 已放弃 legacy `ascendc_library`，改用 CANN 9.0 原生 ASC 构建：

```cmake
find_package(ASC REQUIRED)
project(topk_profile_one LANGUAGES ASC CXX)
add_executable(topk_profile_one kernel_profile.asc)
```

配置时使用：

```text
CMAKE_ASC_RUN_MODE=sim
CMAKE_ASC_ARCHITECTURES=dav-2201
```

`kernel_profile.asc` 由脚本从对应 worktree 自动生成，只保留真实 production 的 FP32 k<=8 device 段：公共 transform、prefix-merge，以及 baseline 的 `dense_repeat_override.h` 或 pivot 的 `pivot_topk_override.h`。host 侧仍然只跑一次指定 shape，并在同一 `.asc` translation unit 内通过 `<<<>>>` 直接 launch kernel。

## 已踩坑

### 1. correctness probe 不能直接拿来 profiling

首版脚本复用 production correctness probe，并在命令后传 `1 1024 8`。但 pivot probe 的 `main()` 固定执行完整 75-case，不读取命令行 shape，所以日志实际从 `E=64,k=2` 开始。

结论：profiling driver 必须只 launch 一次目标 shape。

### 2. `msprof ... finish` 不等于 profiling 成功

child process 失败时，msprof 仍可能打印：

```text
Op profiling finish. Welcome to next use.
```

因此脚本必须检查是否真的生成 `*_instr_exe.csv`；没有 CSV 就立即停止，不运行另一边，也不生成 summary。

### 3. simulator 路径、编译架构、正式 SoC 名是三套概念

原生 ASC 编译使用架构：

```text
dav-2201
```

本机 simulator 文件也落在：

```text
$ASCEND_HOME_PATH/tools/simulator/dav_2201/lib
```

但 AscendC runtime 的 `GetSocVersion()` 识别的是正式 SoC 名，例如：

```text
Ascend910B1 / Ascend910B2 / Ascend910B3 / ...
```

`Ascend920A` 这个 simulator 标签不在它的支持列表里。因此不能把 `Ascend920A` 直接传给 `msprof --soc-version`；否则 msprof 会让 `aclrtGetSocName()` 返回 `Ascend920A`，随后 AscendC runtime 在 launch 前直接判 `UNKNOWN`。

当前脚本保留首参数 `Ascend920A` 仅作为输出目录标签；真正选择 simulator 由 `LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/dav_2201/lib` 完成，并且不再向 msprof 传 `--soc-version`。

### 4. 临时 profiling 源不能带 host tiling/graph 依赖

曾经把 production 文件顶部的 `tiling/tiling_api.h` 一并送进 device precompile，先出现 `register/tilingdata_base.h` 缺失，补 include 后又出现大量 `DT_FLOAT/DT_INT8/...` 宏与 `graph/types.h` 枚举冲突。

这些几十个 error 实际是同一个根因：host tiling/graph 头本来就不该进入 device profiling 源。

当前脚本只抽取：

- 公共常量；
- `StableSoftplusSqrt` / `TransformSelected`；
- `SqrtSoftplusPrefixMergeTopKKernel`；
- baseline 对应 `dense_repeat_override.h`；
- pivot 对应 `pivot_topk_override.h`。

算法正文来自待测 worktree，不维护第二份手写实现。

### 5. `ascendc_library` 的 AIV auto-gen 与当前 `__vector__` kernel 冲突

最小 device 源进入 legacy `ascendc_library` 后，连续三次都卡在同一个位置：

```text
ascendc_kernels_sim_aiv_device
-> auto_gen_kernel_profile.cpp
-> __attribute__((aic)), __attribute__((aiv)), bisheng::core_ratio attribute
   needs to appear with cce kernel func
```

期间尝试过：

```text
__global__ __vector__ void
__vector__ __global__ __aicore__ void
extern "C" __global__ __vector__ __aicore__ void
```

还尝试过移出匿名 namespace。错误阶段和错误文本完全不变，因此这些都没有触及根因。

最终直接查看 CANN 自动生成文件：

```cpp
#undef __global__
#define __global__ inline
#include "kernel_profile.cpp"

#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif
```

关键点是：AIV auto-gen 在 include 原 kernel 源之前主动把 `__global__` 改成 `inline`。所以源码中的 `__vector__` / `__aicore__` 属性最终落到一个普通 inline 函数上，而不是 `cce_kernel`，从而稳定触发同一个错误。

因此前两轮“缺 `__aicore__`”“匿名 namespace/extern C”的判断已被本机日志否定。继续改 kernel 修饰符没有意义。

处理结论：**profiling 不再使用 `ascendc_library`。** 改回 CANN 9.0 原生 ASC 语言构建，让 `.asc` 保持 production 的 `__global__ __vector__` 形式，并通过 `CMAKE_ASC_RUN_MODE=sim` 做仿真编译。

### 6. CPU twin-debug 仍只作为 correctness/API 检查

CPU twin-debug 能验证 API、layout、正确性，但不能证明 simulator/NPU launch 一定成功，也不能作为性能证据。

### 7. clean build

本仓库已有 nested include / 增量依赖跟踪不稳定的历史。profiling 每次都在临时 detached worktree 中 clean configure/build，避免 stale build 影响判断。

### 8. msprof-only launch failure 的根因：错误的 SoC 名注入

原生 ASC sim 二进制直接运行已经通过：

```text
RUN profile-one rows=1 E=1024 k=8 inner=1024
PASS: single-shape profile driver
exit=0
```

`ldd` / ELF RPATH 也确认实际使用了 simulator 的：

```text
libruntime_camodel.so
libnpu_drv_camodel.so
```

显式追加 `LD_PRELOAD=libruntime_camodel.so:libnpu_drv_camodel.so` 后，msprof 行为完全不变，因此 runtime 链接与 preload 都被排除。

随后开启 CANN DEBUG 日志，首次得到决定性错误：

```text
[ERROR] ASCENDCKERNEL ... [GetSocVersion] unsupport soc name: Ascend920A
```

在此之前 runtime 已经正常解析：

```text
.Kernel_Type...
funcType=3
kernelType=2
```

所以 kernel metadata 本身没有问题。

CANN 的 `tools/build/asc_rt/ascendc_runtime.cpp` 中，`GetSocVersion()` 直接读取 `aclrtGetSocName()`，只接受 `Ascend910B*`、`Ascend310P*`、`Ascend950`、`Ascend350`。如果不匹配，就返回 `UNKNOWN`；随后 `AscendLaunchKernelWithHostArgs()` 在不支持的 SoC 分支直接 `return 1`。这与日志中的：

```text
[Bisheng] Launch kernel failure! ret 1
```

完全对应。

根因是 profiling 脚本此前把 simulator 标签 `Ascend920A` 传给：

```text
msprof op simulator --soc-version=Ascend920A
```

msprof 因此把 `Ascend920A` 暴露给子进程的 `aclrtGetSocName()`，导致只有 profiling 模式下 launch 失败；直接运行没有这层错误注入，所以能够 PASS。

当前修复：

- 不再传 `--soc-version=Ascend920A`；
- 使用 `LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/dav_2201/lib` 指定 simulator；
- 保留 `Ascend920A` 仅作为输出目录标签，避免再混淆 simulator 标签与正式 SoC 名。

对应修复提交：`aa90b2f`。尚待本机复验。

## 当前工具

- `tools/profile_topk_costs.sh`：建立 main / pivot detached worktree，生成单形状原生 ASC profiling 源并分别采集；
- `tools/profile_harness/CMakeLists.txt`：原生 ASC sim 构建；
- `tools/profile_harness/main.cpp`：单 shape host correctness harness，由生成的 `.asc` include；
- `tools/diagnose_msprof_launch.sh`：抓取 msprof-only launch failure 的 DEBUG 细节；
- `tools/summarize_topk_profile.py`：聚合 instruction / pipe / code cycles；
- issue #31：记录 profiling 进度、失败原因与最终结论。

## 当前运行方式

```bash
git switch profile-topk-cost-model
git pull
source /home/leo/Ascend/cann-9.0.0/set_env.sh
bash tools/profile_topk_costs.sh Ascend920A 1 1024 8
```

首参数当前只作为输出标签；默认架构为 `dav-2201`。如需显式指定：

```bash
bash tools/profile_topk_costs.sh Ascend920A 1 1024 8 dav-2201
```

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
