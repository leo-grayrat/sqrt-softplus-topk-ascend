# CPU通用TopK相邻路径失败：不计入通过数

新probe曾把E128/k15、E161/k15从sim-only改到CPU。FloatMedium实际生产gate为inner160/k9..16，E128/E161走通用TopK；main和候选均在E128输出0并失败。此类通用路径CPU限制已有记录，不能因为FloatMedium类方法支持更广形状就声称host派发到该类。

修正probe的CPU测试路由后，E128、E161、k17仅由实际设备simulator run_kernel回归。保留旧失败；没有放宽参考/容差，也没有修改生产dispatch。
