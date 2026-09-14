# Round 5 Bisheng 编译修复

第五轮 batched-transform 实现首次在本机 CANN CPU build 时出现编译错误：

```text
non-constant-expression cannot be narrowed from type 'unsigned long' to 'uint32_t'
```

位置为 `dense_repeat_override.h` 中 k=8 contiguous copyout 的 `DataCopyExtParams` 聚合初始化。

根因：`sizeof(float)` 类型为 `size_t`。即使先把 `validRows * VENDOR_KEEP_K` 转成 `uint32_t`，再与 `sizeof(float)` 相乘后，整个表达式仍会按 `size_t/unsigned long` 计算；Bisheng 对 braced aggregate initializer 的 `uint32_t` 字段拒绝这种隐式窄化。

修复：先将完整 byte count 显式收敛为 `uint32_t packedOutputBytes`，再传入 `DataCopyExtParams`。不改变 copyout 数值、布局或第五轮实验设计。

同时结构 checker 增加回归检查，拒绝重新出现 `static_cast<uint32_t>(...) * sizeof(float)` 直接进入 aggregate initializer 的写法。

当前只完成源码修复与静态复核；仍需本机重新执行 Bisheng build 和 CPU correctness，成功后再进入 Judge。