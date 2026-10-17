# C15 批量精确筛选：待完整 Judge

跟踪 [#85](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/85)。分支 `exp/batched-exact-filter-c15`，只需替换 `submission/dense_repeat_override.h`。main仍是PR #84用户确认的15点基线，case15显示1.02 ms；本候选尚无真实NPU/Judge数据。

仅修改FP32 E1024/k64的C15类。batch10、UB176672字节和双输入队列保持原样。每四个原值采样一个，估计mean+1.35*std阈值；随后比较全部原值、精确统计每行候选数。只有所有行都在64..128范围内才走快路径，否则整批走原算法，非有限阈值也回退。阈值只影响命中率，不影响精确Top64。

快路径整批压紧候选，用前缀偏移恢复每行128槽、尾部填负无穷。排序原始FP32值，变换与输出不变；每行Sort32输入512→128值、合并768→128对，无截断或精度降低。静态index/gatherIndex/base/pattern表不覆盖，无新缓冲。本轮将采样、计数和压紧批量化，避免历史阈值方案的逐行同步。

## 实际生产入口 A/B

CANN9/dav-2201/Ascend910B1 simulator，基线固定PR #84后main；同一生产probe、相同输入/核数，多核取最慢核。

| 输入及规模 | 基线us | 候选us | 耗时变化 |
| --- | ---: | ---: | ---: |
| 正态64行/1核 | 54.05 | 31.19 | 减少42.29% |
| 正态1043行/32核 | 34.86 | 21.63 | 减少37.95% |
| 对抗夹具64行/1核 | 59.05 | 63.32 | 增加7.23% |

6次生产仿真输出全部通过。速度依赖输入分布，回退有额外预测/计数开销；不能据模型推算完整case15或排名。65555行及完整15点硬件回归留给CANNJudge。

## 校验与失败记录

- 生产CPU60组通过：默认51组、8种额外分布各1组、正态1043行/32核1组。夹具覆盖64/128候选快路径、63/129回退、尾批及邻近形状。
- 最终研究full/raw各57组，共114组CPU通过；raw检查原始选择值。主机oracle7040行，其中1790快路径行逐位一致、5250行拒绝快路径（拒绝本身不算完整回退输出检查）。
- 完整submission设备编译通过。范围checker确认C15类外与基线一致，研究类去诊断/raw条件后与生产一致；独立只读审查无代码阻塞项。
- v1因FreeTensor需要可变LocalTensor引用编译失败，修正签名。v2正确但正态模型只改善11.69%，uniform/对抗变慢，未准入。v3固定pattern采样在CPU mode3/23行发生PV_MEM写入SIGSEGV、输出不匹配，同环境单独重跑通过，根因未证实。原失败保留，未补COMPLETE或禁用检查器。最终v4改显式uint32采样mask/64元素repeat，通过上述校验；不宣称已证明旧异常根因。

[原始日志、源码摘要](c15-batch-filter-evidence-2026-10-17/summary.json)及同目录manifest记录逐文件SHA256。冻结变更源码与最终header保留，共约0.7MB；公共基线用摘要中的固定main版本复原。研究与实际生产结果分开保存，完整profiles仍在摘要指定WSL运行目录。

先source本机CANN9环境，在新绝对目录复现（编译-j1）：

```bash
bash tools/run_c15_batch_filter_production.sh /home/leo/new-c15-production-run
bash tools/run_c15_batch_filter_research.sh /home/leo/new-c15-research-run
python tools/check_c15_batch_filter_scope.py
python tools/check_c15_batch_filter.py
```

原运行驱动快照保留在证据目录；派生复现脚本已做bash语法检查，未另跑一遍全流程。issue开放等待用户完整Judge，实验未合main。
