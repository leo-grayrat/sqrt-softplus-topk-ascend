# Bulk prefix packing experiment

日期：2026-09-12

分支：`bulk-prefix-pack`

对应 issue：#32。

## 动机

首轮 dav-2201 simulator profiling（`rows=1,E=1024,k=8`）显示：

- baseline `BAR`：127 calls / 25302 cycles；
- `MOV_UB_TO_UB`：42 calls / 672 cycles；
- `VBS32`：562 cycles；
- `VMS4V2`：506 cycles。

因此本实验不再减少 Sort32/MrgSort 数量，而只针对 dense-repeat `PackPrefixes()` 的逐 queue UB→UB copy + barrier。

## 改动

旧实现：每个 queue 单独 `DataCopy`，每次 copy 后 `PipeBarrier<PIPE_V>()`。

新实现：一次 `AscendC::DataCopy(dst, src, DataCopyParams)` 完成所有 queue 的 prefix 压紧，只保留一次 `PipeBarrier<PIPE_V>()`。

参数：

- `blockCount = queueCount`；
- `blockLen = k*2*sizeof(float)/32`；
- `srcStride = (srcStrideFloats-k*2)*sizeof(float)/32`；
- `dstStride = 0`。

当前 dense gate 只覆盖 k=4/8，因此 prefix 长度分别为 32B/64B，均满足普通 DataCopy 的 32B datablock 对齐要求。

## 验证

先运行结构检查：

```bash
python3 tools/check_bulk_prefix_pack.py
```

再做 CPU twin-debug：

```bash
source /home/leo/Ascend/cann-9.0.0/set_env.sh
cmake -S debug/cpu_topk_probe -B /tmp/topk-bulk-prefix-cpu -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build /tmp/topk-bulk-prefix-cpu -j"$(nproc)" --target topk_prefix_production_probe

for rows in 1 3; do
  for E in 128 256 512 1024; do
    for k in 4 8; do
      /tmp/topk-bulk-prefix-cpu/topk_prefix_production_probe "$rows" "$E" "$k" || exit 1
    done
  done
done
```

### 第一次本机验证无效

第一次运行时，本地尚未 fetch 到新分支，先执行：

```text
git switch bulk-prefix-pack
```

得到：

```text
fatal: invalid reference: bulk-prefix-pack
```

随后 `git pull` 仍停留在 `profile-topk-cost-model`，因此 `tools/check_bulk_prefix_pack.py` 也不存在。后续 16 个 CPU twin-debug case 虽全部 PASS，但测试的是旧 production baseline，**不能作为 bulk 实现正确性证据**。

正确切换方式：

```bash
git fetch origin bulk-prefix-pack
git switch -c bulk-prefix-pack --track origin/bulk-prefix-pack
```

若之后本地分支已经建立，则：

```bash
git switch bulk-prefix-pack
git pull --ff-only
```

确认：

```bash
git branch --show-current
git rev-parse --short HEAD
```

正确性真正通过后再做 simulator profile，对比 main 与本分支的 `rows=1,E=1024,k=8`。

## 判定标准

只有当 `BAR` call/cycles 明显下降且没有出现新的大热点时才进入 Judge。若 barrier 热点没有结构性下降，记录负结果并停止，不做 barrier 数量、repeat 参数等微调救火。
