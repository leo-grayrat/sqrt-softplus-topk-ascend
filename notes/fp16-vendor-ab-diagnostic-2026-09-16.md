# FP16 vendor A/B diagnostic

2026-09-16 在 CPU twin-debug 中尝试对同一输入比较新 FP16 vendor path 与原 generic FP16 `run_kernel_base`。

观测：vendor launch 出现 8 个 TmSim/core success；`RUN baseline` 后没有任何 TmSim/core launch，随后 host 读取到 baseline 输出为 0。

结合 `run_kernel_base` 控制流，测试参数满足 shape/dtype/k 前置检查，FP16 generic path 在真正 kernel launch 前唯一可能直接返回的位置是 `BuildTopKTiling(...)`。因此该 CPU-debug 环境下 generic baseline 没有成功进入 device launch，`baseline=0` 不是旧 kernel 的数值结果，不能用于判断 vendor path 语义。

此前 production probe 唯一已见偏差是一个 FP16 rounding-boundary case：设备输出 4，而 host 先用 libm transform 再 cast half 得到 3.998046875；高精度 transform 约 3.999023332，距离两者的 half 舍入中点约 1e-7。production probe 因此改为直接对高精度 transform 检查“半个 FP16 ULP + 很小 FP32 transcendental slack”，避免 host/device 落在舍入中点两侧造成假阳性，同时仍对明显 selection/value 错误保持严格。
