# Gate 1 / Gate 2 实测记录（2026-10-21）

决定是否投入「旁路 ggml 的 MoE 执行器（方案 C）」的两个前置门禁。探针是独立 Vulkan 程序，
不链接 llama.cpp/ggml，不改运行时。

- 源码：[vkgate.cpp](gates/vkgate.cpp)、[gemv.comp](gates/gemv.comp)、[ids.comp](gates/ids.comp)
- 结果：[gate1_boundary_result.txt](gates/gate1_boundary_result.txt)、[gate2_overlap_result.txt](gates/gate2_overlap_result.txt)、
  [gate2_overlap_small_result.txt](gates/gate2_overlap_small_result.txt)
- 设备：AMD Radeon RX 6700S，Vulkan 1.4.315 驱动；compute family=1（非 graphics），transfer family=2（仅 TRANSFER，count=2），
  timeline semaphore 可用。与 ggml-vulkan 的 `prefers_transfer_queue` 条件一致（AMD 非 GCN、非 UMA、未开 graphics queue）。

## 构建与复现

```powershell
cd bench\expert-tiering\gates
glslc -O gemv.comp -o gemv.spv
glslc -O ids.comp  -o ids.spv
g++ -O2 -std=c++17 vkgate.cpp -I"%VULKAN_SDK%\Include" -L"%VULKAN_SDK%\Lib" -lvulkan-1 -o vkgate.exe
.\vkgate.exe info
.\vkgate.exe boundary 500
.\vkgate.exe overlap 1400 200
.\vkgate.exe overlap 400 100
```

## Gate 1：每层「提交 -> 等 router ids -> 回读 -> 下一层」的边界成本

探针每层做：dispatch 一次 14 MiB 权重的流式读（约一层专家字节）+ dispatch ids + 128 B 复制回 host，
然后 host 读回 32 个 ids。用 GPU 时间戳测同一条命令缓冲的纯 GPU 时间，总时间减去它得到边界开销。

| 变体 | 500 次平均 | 相对纯 GPU 工作 |
|---|---:|---:|
| 纯 GPU 工作（时间戳） | 0.206 ms | - |
| A 每次分配+录制+ fence 等待（近似当前调度器） | 0.209 ms | +0.004 ms |
| B 预录制命令缓冲 + fence 等待 | 0.194 ms | -0.012 ms |
| C 预录制命令缓冲 + timeline 信号量等待 | 0.191 ms | -0.015 ms |
| D 深度 2 流水（提交 L+1 后等 L） | 0.115 ms | -0.090 ms |

结论：**本机 Vulkan 的每层边界地板约 0.11-0.21 ms**，而当前 ggml-sched 路径实测是 0.64 ms/split
（`bench/moe-baseline/L2-DECODE-LATENCY.md`）。差距来自调度器的每 split 分配/录制/全设备 drain，不是硬件。
按 32-40 层算，边界总额可从约 20.5 ms 降到 3.6-8.4 ms。

## Gate 2：独立传输队列与计算队列能否重叠

transfer 队列上做 1400 MiB 的 6.7 MiB 分块 H2D；compute 队列上跑 256 MiB 缓冲的流式读（DRAM 受限，避开
Infinity Cache）。两个 command buffer 各自独立预录制，两条队列先提交再等待。

| 用例 | compute 臂 | transfer 臂 | 同队列串行 | 两队列并行 | 隐藏比例 |
|---|---:|---:|---:|---:|---:|
| 1400 MiB / 计算目标 100 ms | 57.2 ms（380 GB/s） | 105.2 ms（13.0 GB/s） | 162.2 ms | **103.6 ms** | 102.8% of min |
| 1400 MiB / 计算目标 200 ms | 90.6 ms（370 GB/s） | 104.8 ms（13.0 GB/s） | 192.8 ms | **105.5 ms** | 99.3% of min |
| 400 MiB / 计算目标 100 ms | 42.2 ms（373 GB/s） | 31.3 ms（12.5 GB/s） | 73.4 ms | **49.1 ms** | 78.0% of min |

结论：**传输与计算在本机确实可以重叠**，短臂的 78%-99% 被隐藏；总时间从「两项相加」变成接近「两者较大者」。
这否证了「本栈不能重叠」的推测，也说明 `bench/moe-baseline/L5-PREFILL-UPLOAD-FLOOR.md` 观察不到重叠的原因是
调度器每次 host-weight 拷贝前的全设备 drain（`bench/moe-baseline/L8-H2D-AND-BANK-DIAGNOSIS.md` §3），
不是驱动或队列能力。

注意：compute 臂显示的 370-380 GB/s 高于 224 GB/s 标称 DRAM 带宽，说明仍有部分命中 Infinity Cache / L2 或
循环被编译器部分优化；这不影响「是否重叠」的判定，但不能用这个数字当作显存带宽结论。

## 对方案 C 收益估算的修订

用 Gate 1/2 的实测值替换原先的「边界 20.5 ms、上传与计算不能重叠」假设，decode 每 token：

```
GPU 数学            约 12 ms（已贴 DRAM 地板，T2-GEMV-CEILING.md §2）
专家上传            约 15.5 ms（206 MiB / 13.9 GB/s），可与 GPU 数学重叠
边界                40 层 x 0.19 ms = 约 7.6 ms（Gate 1 变体 C）
```

| 方案 | 计算 | 估算 |
|---|---|---:|
| 现状（实测） | 37.6-40.5 ms | 26.4 t/s |
| 执行器，上传与计算重叠 | max(12, 15.5) + 7.6 = 23.1 ms | **43 t/s（+63%）** |
| 执行器，冷专家改由 CPU 并发计算 | max(12, 10.6) + 7.6 = 19.6 ms | **51 t/s（+93%）** |
| 上界（边界再压到 0.115 ms/层） | max(12, 15.5) + 4.6 = 20.1 ms | 约 50 t/s（上传仍串行一部分时约 55-60） |

prefill（同上，用实测上传/计算拆分）：

| 配置 | 现状 | 全重叠估算 |
|---|---:|---:|
| pp512（cache 配置） | 383 | 550-680 |
| pp2048 ub2048 | 578 | 780-830 |
| pp2048（LFM2.5） | 874 | 1050-1150 |

TG200 仍不可达：执行器最好约 20 ms/token = 50 t/s；要靠投机解码把一次前向摊到 3-4 个 token 才可能接近
100-150 t/s，而 200 t/s 需要每次前向 <= 12.5 ms（约等于 GPU 数学地板本身）。
