# 多用例合并候选：case7/9/10/15（2026-10-17）

本轮从已获用户完整15/15确认的PR #82后main开始，一次交付四个用例的候选，跟踪[#83](https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/83)。**尚待用户完整CANNJudge，未合入main。** main仍保留case14=251.03 us的已验证版本。本轮不声称新分数或新排名。

仅需替换[submission/dense_repeat_override.h](../submission/dense_repeat_override.h)。没有修改变换公式、输出类型、容差或其他case的生产分支。旧的数值边界问题#52/#68仍单独跟踪。分支`exp/multi-case-selection-c7-c9-c15`保留创建时名称，本轮也包含case10。

## 实现与收益边界

* **case7/9，FP32 E160/k7与E150/k6：** 五个Sort32叶子各保留top8，前四叶只做一次四路合并，再与第五叶的top8合并；merge产物从80降至48对/行，移除虚设的第二组合并。batch仍48。复制调用6→5，但复制量56→96对，不能描述为搬运量也减少；含merge的相关UB写入1600→1152字节/行。新增逻辑只在原inner160且k6/7入口，邻近填充形状也验证。最终结果在pairB，返回false与调用方选择一致。
* **case10，FP16 E128/k16：** 把相隔64列的两值组成64组，按组最大值精确选16组，保留这16组的两个原值，再选top16。Sort32输入128→96值，merge64→32对。任意落选组的最大值前已有16个入选组最大值，因此不丢top16分数；相同分数保留正确重数。仅匹配E128/k16。UB最大149024字节。
* **case15，FP32 E1024/k64：** 复用不同时段的maxima/packed/tmp和maxA/maxB/pairA/pairB，保留独立索引表与双输入队列，batch8→10。四个连续64值segment、每行一个repeat，把大batch的Max主机调用从3×rows降至12次。UB为17664×batch+32，最大176672字节。选择算法、候选数与变换不变。连续batch模型仅改善约3.5%–3.7%，作为本轮附带收益，不再为这一小改进单独要求上传。

case10首版batch48让4096行/32核的忙核处理144行（基线128行），抵消收益；首版模型14.96→14.86 us，仅0.67%，原样归档。最终用`ceil(perCore / ceil(perCore/48))`平衡batch：perCore128时batch43、忙核129行，16384/32时batch47、忙核517行（首版528行）。最终4096行模型13.71 us。主机6162组分配检查及完整16384行CPU校验通过；完整16384行设备仿真未运行。

## 同一harness的模型A/B

CANN9、dav-2201、Ascend910B1 simulator，取各核`duration_time(us)`最大值。基线固定PR #82后main；C7/9/15取candidate-v2（这些类与最终v4逐字一致），case10取最终v4。所有下表运行同时检查输出。模型时间不能直接当作真实Judge时间或按比例预测分数。

|形状（r行数/e列数/k/c核数）|基线/us|候选/us|耗时减少|
|---|---:|---:|---:|
|r1-e1024-k64-c1|3.99|3.58|10.28%|
|r1-e150-k6-c1|4.98|4.60|7.63%|
|r1-e160-k7-c1|4.98|4.60|7.63%|
|r1043-e1024-k64-c32|39.51|38.05|3.70%|
|r192-e150-k6-c1|27.97|22.90|18.13%|
|r192-e160-k7-c1|27.11|22.05|18.66%|
|r32-e1024-k64-c1|32.17|31.00|3.64%|
|r385-e150-k6-c8|11.69|10.10|13.60%|
|r385-e160-k7-c8|10.25|9.00|12.20%|
|r64-e1024-k64-c1|61.22|59.05|3.54%|
|r96-e150-k6-c1|16.03|13.50|15.78%|
|r96-e160-k7-c1|15.34|12.82|16.43%|
|r1-e128-k16-c1|3.89|3.45|11.31%|
|r192-e128-k16-c1|20.85|18.58|10.89%|
|r4096-e128-k16-c32|14.96|13.71|8.36%|
|r49-e128-k16-c8|4.01|3.89|2.99%|
|r96-e128-k16-c1|11.81|10.86|8.04%|

其他对照只有候选输出检查，未作同形状基线A/B，因此不宣称性能改善。v3的联合C15检查使用FP32 scratch probe，与C15 group probe输入不同，不能跨harness比较耗时。单次模型差异尚无重复运行置信区间；完整15点的其他路径回归和真实收益交给Judge。

## 本地验证与源码对应

* 主机精确选择：C7/9共602112行、3913728个选中分数位值；C10共9408行、150528个位值，覆盖batch1–48及8类分布；C15检查56320个Max地址映射。主机检查不能替代设备执行。
* 基线和FP32候选v2：各87组FP32小k生产入口、51组C15生产入口及9个C15边界CLI用例，共各147组CPU通过。候选v2完整device构建通过。
* C10基线：新half生产probe 29组CPU及5次设备仿真通过。联合v3：29组half、87组FP32及C15行23，共117组CPU；11次设备仿真、完整device构建通过。
* **最终v4：29组half CPU，加43/48/128/16384行的4组CLI CPU，8次设备仿真及完整device构建通过。** 与v3只差C10的batch计算两行，FP32两类与v2/v3逐字相同；最终header与v4冻结构建输入字节相同。所有成功版本合计53次设备仿真，分版本归档，不把中间版本全部冒称最终header验证。
* probe调用实际生产dispatch/run_kernel，变换参考和half容差沿用已有生产probe，没有放宽。默认CPU不走已知会触发#9检查器限制的k17通用TopK；k17作为设备sim对照通过。C14 E512/k10、C12、C8及临近形状有回归覆盖。新half probe没有启用继承的VALUES_ONLY_RAW宏，不能声称raw设备probe通过。
* 独立审查无Critical/Important问题；scope guard将四处准入变更还原后，整个生产header必须与基线相同，保障case14及其他实现未混入修改。

## 失败与未采用方向

cap11首版UB194336字节，默认51组C15和单独10行CPU通过，但单独11行在Gather<int>的PV_MEM读路径SIGSEGV并输出错误。静态范围检查未发现越界，故不宣称硬件有32KiB限制，也未关闭检查器。首版失败日志、源文件和未完成driver保留；不标COMPLETE，不计入成功仿真。最终cap10的全部边界9/10/11/12/19/20/21/22/23行通过。详见[语法与模型问题记录第14项](syntax-errors.md)。

曾考虑用跨bank stride把case15 Max压成3次调用。官方[A2/A3 220x避免bank冲突](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00020.html)指出16倍block stride会把单指令多个块集中到同一bankgroup；本轮不实现该未经测量假设，采用块步长1的12次方案。API依据仍为本地固定CANN9 Max/MrgSort/Gather文档及[CANNBot核对记录](cannbot-resources-2026-10-16.md)。case8旧负结果没有重新包装为新优化。

## 复现与证据

[模型全部核时间、CPU摘要及版本对应](multi-selection-evidence-2026-10-17/model-summary.json)；[冻结输入和原日志SHA256清单](multi-selection-evidence-2026-10-17/sha256-manifest.json)；[独立审查](multi-selection-evidence-2026-10-17/review.md)。日志和CMake依赖的冻结src按原字节保存，不整理换行。清单覆盖收集时450个原始文件，主机JSON、审查和派生summary另列文件，不在该原始清单内。巨型trace/指令CSV留在WSL原构建目录，不入库。

```bash
source /home/leo/Ascend/cann-9.0.0/set_env.sh
python3 tools/check_multi_c79_selection.py
python3 tools/check_multi_c15_layout.py
python3 tools/check_multi_c10_selection.py
python3 tools/check_multi_selection_scope.py
# 每次使用不存在的新绝对路径，编译-j1；不要覆盖归档中的旧目录。
bash tools/run_multi_selection_probe.sh baseline /home/leo/repro-multi-baseline-new
bash tools/run_multi_selection_regression.sh baseline /home/leo/repro-multi-baseline-new
bash tools/run_multi_selection_probe.sh candidate /home/leo/repro-multi-candidate-new
bash tools/run_multi_selection_regression.sh candidate /home/leo/repro-multi-candidate-new
bash tools/run_multi_c10_probe.sh baseline /home/leo/repro-multi-half-baseline-new
bash tools/run_multi_c10_balanced_probe.sh /home/leo/repro-multi-final-new
```

candidate命令重建当前最终版本，历史cap11/v2/v3由各归档`src`与`repro-snapshot.sh`复现；历史源码不能用当前header替代。单独完整16384行CPU命令见[附加驱动](multi-selection-evidence-2026-10-17/run-extra-v4-cpu.sh)，修改构建路径后运行。构建环境为WSL Ubuntu22.04/CANN9，无本地物理NPU。

下一步只有用户上传同一header，运行完整15点CANNJudge。收到结果后记录同次全部时间，决定是否经PR合并；#83在此之前保持开放。
