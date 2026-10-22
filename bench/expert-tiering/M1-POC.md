# M1：MoE 执行器 PoC 结果（2026-10-21）

源码：[pocs/moe-exec/moe-exec.cpp](../../pocs/moe-exec/moe-exec.cpp)。独立程序，不接运行时，用真实 GGUF 权重。

## 方法

- 读取 Qwen3.6 第 N 层：`ffn_gate_inp.weight`（router，F32 [2048,256]）、`ffn_{gate,up,down}_exps.weight`
  （Q4_K/Q6_K，256 专家），全部上传到一个 Vulkan backend buffer（约 434-500 MiB）。
- 每层两个子图：
  - router：`mul_mat -> soft_max -> top_k(8)`，输出 ids；
  - experts：`mul_mat_id(gate) / mul_mat_id(up) -> silu*mul -> mul_mat_id(down) -> 加权 -> sum_rows`。
- 三种测量：
  - A 边界：提交 router + 计算 + ids 异步回读 + 同步（执行器每层必须付的部分）；
  - B 专家计算：每层提交 expert 子图并同步；
  - C 流水：A 之后立刻提交 expert 子图、不与下一层同步（执行器实际形态）。
- 每层都换用两套图张量（双缓冲），避免复用同一 ids 张量造成的假串行。

## 结果（RX 6700S，40 次迭代，均稳定复现）

| 层 | A 边界 ms/层 | B 专家 ms/层 | C 流水 ms/层 | 边界 vs 0.64 ms/sched |
|---:|---:|---:|---:|---:|
| 10 | 0.185 | 0.361 | 0.512 | 3.5x |
| 20 | 0.160-0.202 | 0.354-0.446 | 0.516-0.566 | 3.2-4.0x |
| 35 | 0.173 | 0.384 | 0.579 | 3.7x |

验收：边界 <= 0.25 ms **通过**；与 sched 0.64 ms/split 相比 >= 2x **通过**（3.2-4.0x）。

## 结论与对 M2 的含义

1. **边界不是问题**：每层 router 计算 + ids 回读只要 0.16-0.20 ms，与 Gate 1 的 0.19 ms 一致；
   当前调度器每 split 0.64 ms 的差距来自 split 分配/录制/全设备 drain，执行器可以拿掉。
2. **每层 MoE 专家计算的固定成本约 0.35-0.38 ms**，主要随 kernel 数走，不随专家数走：
   n_used=1 与 n_used=8 的 B 几乎相同（0.333 vs 0.354）。把加权求和从 8 个 add 改成
   permute+`sum_rows` 后，B 从 0.446 降到 0.354（-21%）。
3. 按 C 的 0.52-0.58 ms/层，MoE 部分约 **20.6-23 ms/token**。要达 TG 43（23 ms/token 总量），
   M2 必须让 attention/dense（约 12 ms）与 MoE 的上传/计算**重叠**，不能是串行相加；
   同时继续压 MoE 每层 kernel 数（合并 gate/up、融合 silu*mul 需要重排或新 kernel）。
4. B2（40 层连续 async、只在最后同步）实测 0.9-1.0 ms/层，**不是有效指标**：复用同一组张量存在数据冒险，
   后端插入了等待；执行器必须用每层独立/双缓冲的目的地。不要把 B2 当作 GPU 纯计算时间。

## 复现

```powershell
cmake --build build-vulkan --target llama-moe-exec -j 8
build-vulkan\bin\llama-moe-exec.exe F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf 20 40
```

注：`pocs/` 只在 `LLAMA_BUILD_EXAMPLES=ON` 时进入构建；本构建目录为启用该目标已把 `LLAMA_BUILD_EXAMPLES` 设为 ON
（只影响配置，不影响已有二进制）。

## M2 风险探针：CPU 与 GPU 能否真正并发（llama-moe-hybrid）

方案 C 的 43 t/s 依赖“CPU 算冷专家、GPU 算热专家同时进行”。图内 hybrid 是串行的（3x 慢），
所以先用独立探针验证硬件与驱动层面能否并发。

源码 [pocs/moe-exec/hybrid.cpp](../../pocs/moe-exec/hybrid.cpp)：GPU bank 4 个专家、CPU bank 4 个专家（同一层真实权重），
CPU 用**常驻线程**（避免每次 new thread 的假象），GPU 用 async compute + synchronize。

| 测量 | 结果 |
|---|---:|
| CPU 4 专家 | 0.575 ms |
| GPU 4 专家 | 0.330 ms |
| 串行和 | 0.905 ms |
| 并发 | **0.639 ms** |
| 隐藏 | 0.266 ms = min(cpu,gpu) 的 **80.4%** |

注意：首次版本每次 `std::thread` 新建线程，并发反而 1.283 ms（比串行差）；改成常驻线程后才是真实结果。
结论：**CPU/GPU 并发在本机成立**，M2 的 H 模式有硬件基础；实现时必须复用线程池，不能在每层创建线程。

