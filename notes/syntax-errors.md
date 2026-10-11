# CANN / Bisheng 编译错误速记

调试约定：每发现一个明确问题，先建 GitHub issue；修复后保持 issue 打开，直到本地或 CANNJudge 重新验证该问题不再出现，再关闭。

## 2026-10-11：FP16 CPU 模板 kernel 注册限制（#64）

直接调用真实 run_kernel 时，main 和候选均报 `Kernel mode not found ... sqrt_softplus_top_k_dispatch_custom<half>`。现有 FP16 probe 已用非模板包装绕开同一限制。仅 CPU 构建沿用包装方式调用生产 RunVendorTopKDispatch；simulator 仍直接 run_kernel，检查 host tiling 和实际模板 launch。生产源码不因调试器改变，不把 CPU 包装结果写成完整 host 入口验证。

包装 global 的定义也必须对 device 编译可见，不能整体放入 `#ifdef ASCENDC_CPU_DEBUG`，否则生成 mode registry 时找不到它，同样报错。只对调用位置条件编译，沿用既有 probe 的全局定义布局。

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

## 6. `DataCopy(LocalTensor, LocalTensor, count)` 的搬运字节数必须 32B 对齐

症状：CPU checker 报：

```text
Failed to check count value in DataCopy from LocalTensor to LocalTensor,
count * sizeof(T) must be 32B align.
In NPU mode ... rounded down by 32B.
```

当前实例：prefix-merge probe 最终把前 `k` 个 `(score,index)` pair 暂存到输出队列时，按 `k*2` 个 `float` 调用基础 `DataCopy`。`k=2` 时只有 4 个 float = 16B，CPU 直接报错，NPU 语义会向下取整为 0B。

处理：LocalTensor 间基础搬运按 32B 粒度向上取整；真正写回 GM 时仍可用 `DataCopyPad` 按实际 `k*2*sizeof(float)` 字节输出。当前修复见 issue #23，已由用户本地完整多层 probe 验证通过。

## 7. `MrgSort` 不能用 `validBit=1` 表示单条有效队列

症状：CPU checker 报：

```text
Failed to check validBit value in MrgSort, its valid value is 3 / 7 / 15
```

当前实例：prefix-merge tree 某层队列数为 `4n+1` 时，最后一组只有 1 条输入；原逻辑按 `(1<<active)-1` 得到 `validBit=1`。`E=129` padding 到 `inner=160` 后有 5 个 Sort32 块，第一层正好稳定触发。

处理：单条剩余队列不调用 `MrgSort`，直接把其前 `k` 个 pair 按 32B 对齐透传到下一层槽位；只有 `active=2/3/4` 时才调用 `MrgSort`，对应 validBit 3/7/15。见 issue #25，等待用户本地重新运行 padding probe 验证。
## 8. AICore 标量 unsigned integer 不能直接转换为 float

2026-10-04，#48 整行压缩 probe 编译中，`static_cast<float>(count)`（count 为 uint64_t）被 Bisheng 拒绝：`cast between floating and unsigned integer variable is not allowed in aicore function`。

该计数在实验中严格限定为 0..1024，先转换为 int32_t，再转换为 float 以写入诊断输出即可编译，CPU 检查通过。这个修复依赖已知范围，不能推广到任意 uint64_t 值。生产提交没有此转换。

## 9. 通用 TopK 空索引张量的 CPU 检查器限制

2026-10-04，#49 基线 probe 调用真实 run_kernel 时，CPU 环境需先通过 `PlatformAscendCManager::GetInstance("Ascend910B1")` 初始化平台，否则 host tiling 可能直接失败并未启动 kernel。

初始化后，原有通用 TopK 路径被 CANN 9 CPU 检查器拒绝：未使用的默认 `LocalTensor<int32_t> srcIndexLocal` 被识别为 GM，报 `Failed to check tensor position of srcIndexLocal`。没有禁用检查器或修改旧提交；旧路径改用 simulator 验证，32/64 行均通过。新增 medium-k 路径绕过高阶 TopK，CPU 31/31 通过。不要把这个 CPU 失败描述成已有 Judge 版本在真机计算错误，也不要声称基线 CPU 通过。
