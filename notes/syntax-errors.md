# CANN / Bisheng 编译错误速记

调试约定：每发现一个明确问题，先建 GitHub issue；修复后保持 issue 打开，直到本地或 CANNJudge 重新验证该问题不再出现，再关闭。

## 1. host 函数不能直接从 `__aicore__` 调用

症状：

```text
candidate function not viable: call to [host] function from [aicore] function
```

当前实例：`AlignUpBytes()` 是普通 host `constexpr` 函数，却在 `SqrtSoftplusTopKKernel::Init` / `ProcessRow` 这类 `__aicore__` 函数里调用。

处理：让该计算在 aicore 侧可用，或直接在 aicore 函数内写对齐算式。见 issue #2。

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

处理：对最终结果显式转成 `uint32_t`，或先算进一个明确的 `uint32_t` 变量。见 issue #3。

## 3. CANNJudge 参赛代码不要加入 stdio 调试输出

症状：还没进入编译就提示“提交代码中存在不合规内容”。

当前实例：`kernel.asc` 中加入 `<cstdio>`、`std::fprintf`、`stderr` 后触发；删除后已由云端验证可进入编译。

处理：参赛 `kernel.asc` 不使用这些调试输出。见已关闭 issue #1。
