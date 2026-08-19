# 性能与正确性验证约束

## 固定变量

开始比较前固定 commit、AMD driver、Vulkan SDK、模型文件/hash、量化、context、threads、batch/ubatch、device、build flags、MTP 状态和重复次数。不同变量的结果不能直接比较。

## 最小矩阵

按机器能力逐步执行：

- Q4_K_M 和 Q3_K_M；
- context 4K、8K、16K、32K；
- PP 16、64、256、512、1024；
- TG 1、32；
- MTP OFF、N=1、N=2、N=3。

## 必须记录

端到端记录 TTFT、PP/TG tok/s、P50/P95/P99 token latency、VRAM/RAM、CPU/GPU 使用率、失败和 fallback。Expert 优化额外记录 cache hit/miss、resident 数、upload bytes/latency、prefetch hit、prediction accuracy、`MUL_MAT_ID` 时间和 dispatch 数。

## 质量门

- 保留 upstream fallback，并在同一 workload 下对比。
- 运行可用的 `test-backend-ops` 和相关 CTest；模型相关测试注明模型和设备前置条件。
- 用 `llama-perplexity` 或等价固定 prompt/greedy 输出做质量回归。
- 保存 `llama-bench -o json`/`-o csv` 原始结果和命令行，不以理论带宽或预计 tok/s 代替测量。
- MTP acceptance rate 不能单独代表 net throughput；必须计入 draft 和 verify overhead。
