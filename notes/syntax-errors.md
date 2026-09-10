# CANN / Bisheng 编译错误速记

调试约定：每发现一个明确问题，先建 GitHub issue；修复后保持 issue 打开，直到本地或 CANNJudge 重新验证该问题不再出现，再关闭。

## 1. host 函数不能直接从 `__aicore__` 调用

症状：

```text
candidate function not viable: call to [host] function from [aicore] function
```

当前实例：`AlignUpBytes()` 是普通 host `constexpr` 函数，却在 `SqrtSoftplusTopKKernel::Init` / `ProcessRow` 这类 `__aicore__` 函数里调用。

处理：让该计算直接在 aicore 侧完成，不跨 host/aicore 调用。

验证：2026-09-10 用户本地使用 CANN 9.0 / Bisheng 重新编译，原报错消失并成功链接。见已关闭 issue #2。

## 2. `sizeof(T)` 会把整数表达式提升为 `size_t`

症状：

```text
non-constant-expression cannot be narrowed from type 'unsigned long' to 'uint32_t'
```

当前实例：

```cpp
static_cast<uint32_t>(k_) * sizeof(T)
```

虽然 `k_` 被转成了 `uint32_t`，但 `sizeof(T)` 是 `size_t`，整个乘法仍变成 `size_t`；直接放进 `uint32_t` 字段会被 Bisheng 的列表初始化检查拒绝。

处理：先把最终字节数明确算成 `uint32_t`，再传给 `DataCopyExtParams`。

验证：2026-09-10 用户本地重新编译，原报错消失并成功链接。见已关闭 issue #3。

## 3. CANNJudge 参赛代码不要加入 stdio 调试输出

症状：还没进入编译就提示“提交代码中存在不合规内容”。

当前实例：`kernel.asc` 中加入 `<cstdio>`、`std::fprintf`、`stderr` 后触发；删除后已由云端验证可进入编译。

处理：参赛 `kernel.asc` 不使用这些调试输出。见已关闭 issue #1。

## 4. 官方模板在 host 侧强转 `GM_ADDR` 会产生 `cce_global` 警告

症状：

```text
warning: 'cce_global' attribute ignored when parsing type [-Wignored-attributes]
```

位置：starter project 自带的 `main.asc` 中：

```cpp
run_kernel((GM_ADDR)inputDevice_router_logits, ..., (GM_ADDR)outputDevice_topk_scores, ...);
```

原因：`GM_ADDR` 展开为带 `__gm__` / `cce_global` 属性的指针类型，而这行代码位于 host 侧。Bisheng 在解析 host 侧强制类型转换时忽略该设备地址空间属性，于是给出 warning。

判断：这行代码在官方空工程里原本就存在；当前编译仍成功完成链接，所以它不是本项目 kernel 的编译失败原因，也暂不修改官方模板。见 issue #6（当前保留为已知问题）。

## 5. CPU 孪生调试的 `<<<>>>` 需要 CPU launch 支持头

症状：

```text
use of undeclared identifier no AscCPUKernelLaunch found under AscendC namespace or multiple found
```

当前实例：`debug/cpu_topk_probe` 以 `CMAKE_ASC_RUN_MODE=cpu` 编译时，probe 自身以及被 include 的 `submission/kernel.asc` 中所有 `<<<>>>` launch 一起报错。

原因定位：重新核对 CANN 9.0 当前官方样例后，CPU debug 并非一律禁止 `<<<>>>`；官方样例会在 `ASCENDC_CPU_DEBUG` 下包含 `cpu_debug_launch.h` 以提供 CPU launch 支持。probe 最初漏了该头文件。

处理：仅在 probe 中加入：

```cpp
#ifdef ASCENDC_CPU_DEBUG
#include "cpu_debug_launch.h"
#endif
```

参赛 `submission/kernel.asc` 不改。见 issue #12；等待用户本地重新编译验证后再确认关闭。
