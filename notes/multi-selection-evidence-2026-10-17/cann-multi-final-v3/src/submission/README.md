# CANNJudge starter project

这里保存比赛官方提供的空工程，并以它作为唯一的提交工程骨架。

原始压缩包：`sqrtsoftplustopk_problem_1744_template(1).zip`

SHA-256：`dc668c71ad672034ced90d1c834d2e9cab8d23fd00948251e27f65876ae89441`

当前这批文件为原始模板内容，没有对 `kernel.asc` 等比赛文件做实现修改。

已确认：

- CMake 默认 `NPU_ARCH` 为 `dav-2201`；
- 模板采用 Ascend C Kernel 直调方式；
- 本地公开用例只有 case0：`T=8, E=64, k=4, float32`；
- 模板自带 Python golden 会把最终输出转回输入 dtype；
- 当前 ChatGPT 执行环境没有 CANN，因此这里尚未进行 Ascend C 编译或 NPU 运行验证。
