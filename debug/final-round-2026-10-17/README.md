# 最后一轮研究归档（未接入生产）

所有补丁的目标均为`submission/dense_repeat_override.h`，禁止将这些补丁当作已Judge通过的生产版本。详见[复盘](../../notes/contest-retrospective-2026-10-18.md)和[冻结结果](../../notes/final-round-evidence-2026-10-17/summary.json)。

1. `c8-c11-local-tested.patch`相对于PR #84后main（b8bebe），有152组CPU、设备编译和模型输出检查，收益不足、无Judge。
2. `c14-cap28-local-tested.patch`相对于应用第一项后的冻结源码，有独立37组CPU和模型检查，未进行完整联合设备编译或Judge。
3. `c7-c9-UNVALIDATED.patch`相对于应用第一项后的冻结源码，与第二项互相独立。未编译、未测量、未Judge。

`.sh`为当时实际驱动/待启动脚本，包含原工作目录和固定绝对路径，不是对当前main的幂等安装程序。`final-c79.sh`从未成功启动。复现需新建独立目录，先按对应依赖应用补丁，并重新核对路径和基线；不要在生产工作目录直接运行或覆盖旧证据。
