# CANN官方与社区资料的使用（2026-10-16）

用户提醒后，本轮已实际拉取并阅读[CANNBot Skills](https://gitcode.com/cann/cannbot-skills/tree/master)与[asc-devkit v9.0.0](https://gitcode.com/cann/asc-devkit/tree/v9.0.0)。只作为工作区参考，没有全局安装或替换比赛starter工程。CANNBot固定读取版本fe65c77485d42a93c7e45f3e67ab96d95504b9dc；devkit固定v9.0.0标签c8c1b9df6bb9cd61c09751c7ef414b89e06af274。路径与默认版本会随仓库演进变化。

## 用在哪里

- [ascendc-docs-search](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-docs-search/SKILL.md)：按API索引定位，列出所有同名及数字后缀变体，逐项核对函数签名、芯片支持、单位与限制。本轮查看Duplicate及Duplicate-49、Gather及Gather-51、三种MrgSort变体，防止把950寄存器API当910B接口。
- [ascendc-api-best-practices](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-api-best-practices/SKILL.md)：读了归并排序、buffer、transpose/Gather及repeat限制参考。核对四路各自有序、缓冲生命周期、双缓冲与队列深度的区别。生产仍使用既有CANN9接口，不套用大规模跨核全排序模板。
- [性能优化技能](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-perf-optimize/SKILL.md)及单核流水诊断：先做UB/工作量建模和同协议测量，区分向量、标量、访存或流水不足；本算子无跨卡通信和跨核同步，跳过这两类分析。尚未据聚合耗时声称某个PIPE屏障单独花了多少时间。
- [同步检查技能](https://gitcode.com/cann/cannbot-skills/tree/master/ops/ascendc-sync-audit)：后续同步审查的参考入口。本轮没有声称运行其审计脚本，也没有因为添加这项资料而删减生产屏障。

## 已经核实的接口事实

[Duplicate的v9文档](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/docs/api/context/Duplicate.md)说明scalar mask是连续元素数，位掩码是数组形式。float连续mask为1–64。本轮误用65535作为低16位掩码的问题，本应在编码前据此排除。

[基础MrgSort的v9文档](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/docs/api/context/MrgSort.md)明确repeat生效需要四路等长、连续存储、validBit=15且不启用exhausted suspension。该约束直接解释首版三路跨行合并不能成立。最终使用四份等长12-pair列表，第四份为-inf填充。

[Gather的v9文档](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/docs/api/context/Gather.md)规定偏移单位是字节，连续mask与每次迭代容量依赖元素宽度。本轮最终按uint32原样搬运相邻两个half，地址4字节对齐，经独立主机位值检查、CPU逐值选择与完整变换检查验证，没有将half转成整数数值来比较。

## 适用边界

实际读取的docs-search技能默认环境写着“A3、CANN8.5.0”，目录提示是`docs/zh/api`。本机是CANN9.0、dav-2201、Ascend910B1 simulator；v9.0.0发布资料实际在`docs/api/context`。因此先读真实Git树和产品支持表，再用本机SDK核对，实现与编译测试优先于技能中的默认环境。

master中的RegBase、SIMT、VF等资料可能只适用于950/arch35；不能据“CANN官方技能”就直接应用。`ascendc-topk.md`是高频代码问题清单，名称不代表提供本题Top-K的专用高性能算法。Windows资料checkout有一次git-lfs过滤器错误，所读Markdown可正常访问，未把缺失图片或未读文件算成已审查资料。

这些资料能明显减少API误用、提供性能假设与验证方法；能否改善本题仍由匹配环境的编译、输出检查及用户Judge确认。技能不替代NPU性能证据。
