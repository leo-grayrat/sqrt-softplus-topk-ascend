# dense-repeat production rebuild note

`submission/kernel.asc` 通过 `#include "dense_repeat_override.inc"` 间接包含 dense-repeat 实现。2026-09-12 调试中发现，仅修改 `.inc` 后直接执行既有 build 目录的 `cmake --build build`，CPU twin-debug 的报错调用地址与修改前保持一致，疑似 ASC/CMake 依赖没有追踪该嵌套 include，导致旧对象未重编译。

验证 dense-repeat 修改时应先执行 clean configure/build：

```bash
rm -rf build
cmake -S . -B build -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build build -j"$(nproc)"
```
