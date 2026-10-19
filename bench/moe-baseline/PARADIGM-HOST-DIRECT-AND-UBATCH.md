# 新权重访问范式评估：ubatch 流水 vs PCIe 直读（2026-10-19，task-19，最终阶段）

> 分支 `moe-gpu-cpu-hybrid` @ `a847838db`。**最终结论：两种范式在本机都不可行/达不到验收线，
> 未保留任何引擎改动**；现有配置就是这台 8 GiB 机器的上限。终测数据在第 5 节（与 task-18 一致，±3%）。

## 1. 长 prefill 时间分解（实测，10k prompt = 10428 tokens）

配置：`-ncmoe 41 --moe-cache-mib 5120 -ub 2048 -c 32768 -t 5 -fa on`，prefill 408-411 t/s。
用 `GGML_SCHED_TRACE=1` + `GGML_MOE_COPY_TRACE=1` 统计（perf logger 在长 ctx 下会崩，见第 6 节）：

| 项 | 每 ubatch(2048 tok) | 10k prompt 全程 | 占比 |
|---|---:|---:|---:|
| host->device 专家上传字节 | ~14.0 GiB | 69.6 GiB (71278 MiB) | - |
| 上传时间（13.9 GB/s 天花板） | ~1.08 s | ~5.4 s | **21%** |
| GPU 计算（MoE GEMM + attention/SSM + dense） | ~4.0 s | ~20.1 s | **79%** |
| 其中 bank D2D 拷贝 | 1.5 GiB | 7.4 GiB | ~0.1 s |
| 调度器 copy 记账 | issue 20 ms + sync 12.5 ms | - | 小 |

结论：**最大的单项是 GPU 计算（79%），上传只占 21%**；任何"隐藏上传"的范式最多只能拿回 21%
（PP 理论上限 +27%，但要全额隐藏才够；实际预测命中率会打折）。

## 2. P1 ubatch 级流水：收益上限低于验收线，且缺显存

设计：ubatch k 计算时，用 ubatch k 的 routing 作为预测，把 ubatch k+1 同层专家上传到第二个
scratch/bank；ubatch k+1 的拷贝命中就不需要再传。
- 收益上限：隐藏 21% 上传 -> PP +27%（理论）；但预测只在相邻 ubatch 内容相似时命中，
  实际可隐藏的可能是 50-70% -> **PP +10-15%**，命中低时更低。
- 显存：需要 ~1 层专家（~350-500 MiB）的第二缓冲；实测 256K ctx 峰值已 7956/8176 MiB、
  32K cache5120 峰值 7752/8176 MiB，**没有空间**，必须从 cache 里让出（cache 少 512 MiB
  约 -1.5~2% PP），净收益进一步缩水。
- 另有一个结构性限制：预取目标是"下一 ubatch 的预测集合"，而 ubatch k+1 的 routing 要等
  ubatch k 算完才出现，所以预取必须在 ubatch k 的计算期间完成（~4 s 内传 ~14 GiB 的 60-70%，
  即 8-10 GiB / 13.9 GB/s = 0.6-0.7 s 可行，链路不是问题；命中率才是）。
- 判定：**实现成本高（scheduler+context 的预取/命中/回退）、净收益 < 验收的 +15%、且需要压缩
  cache**，不作为本轮落地项。

## 3. P2 GPU 直读 host 权重（VK_EXT_external_memory_host）：本机 AMD 驱动不可用

- 代码事实：`ggml_vk_host_buffer_import_usable()` 对 AMD 明确返回 false，注释
  "amdgpu re-faults all pages of every imported host allocation on each submission" --
  即 AMD 驱动每次提交都会重新缺页所有导入的 host 分配，后端作者因此禁用了 GPU 访问 host 内存。
- 直接实验（env 门控，已回退）：把 Vulkan 的 host buffer type 加入
  `ggml_backend_vk_device_supports_buft`，让 MUL_MAT_ID 直接读 host VkBuffer（专家权重本就在
  Vulkan_Host 的 pinned VkBuffer 里，理论上设备可寻址）：**立即崩溃**（exit 0xC0000005，
  无任何日志输出），说明当前 Vulkan 后端在描述符/指针路径上并不支持 host buffer 作为 shader 输入；
  修好它需要逐条审计后端 buffer/descriptor 路径，并且仍要面对上面那条 AMD 驱动的重缺页行为。
- 判定：**在本机不可行**（不是参数问题，是驱动+后端设计）。

## 4. P3 融合 streaming MoE GEMM：分析结论 = 需要 P2 的能力，同样被堵

- 思路：K-tile 级流式读 host 权重并累加，彻底取消显式拷贝。
- 但"shader 直接读 host 权重"正是 P2 被 AMD 驱动禁掉的能力；没有它，streaming 只能走
  "拷贝到 staging 再算"，还是回到 21% 上传。
- 若能用：上传 21% 可完全隐藏 -> PP +27%（上限），代价是新的 Vulkan kernel + 布局/正确性工作。
- 判定：**报告即可，不实现**（被 P2 的前置条件堵住）。

## 5. 最终判据测试（无引擎改动，r=1-2，与 task-18 ±3% 内一致）

| 场景 | 配置 | prompt | prompt-eval | decode@depth | 峰值 VRAM |
|---|---|---|---:|---:|---:|
| 256K ctx | `-ncmoe 41 -c 262144 -ub 2048 -t 5` | 10428 | 371.8 t/s | 14.2 t/s | 7956 MiB |
| 256K ctx | 同上 | 19758 | 320.7 t/s | 14.2 t/s | 7956 MiB |
| <=32K ctx（推荐长上下文） | `-ncmoe 41 --moe-cache-mib 5120 -ub 2048 -c 32768` | 10428 | 410.6 t/s | 21.5 t/s | 7752 MiB |
| <=32K ctx | 同上 | 19758 | 346.9 t/s | 20.3 t/s | 7752 MiB |
| 短 prompt（推荐通用） | `-ncmoe 41 --moe-cache-mib 6144 -ub 2048` | pp512 383.2 / tg32 26.19 | - | - | 8028 MiB |

与 task-18 基线（363.4 / 309.9 / 401.3 / 333.9 / 386 / 26.2）差异在 ±3% 漂移内，**无回归**。

## 6. 测量工具备注

`GGML_VK_PERF_LOGGER=1` 在长 ctx（32K）下会崩溃（exit 0xC0000409，perf logger 的 buffer 假设
在长序列上不成立），所以本轮的分解用 `GGML_SCHED_TRACE` + `GGML_MOE_COPY_TRACE` 的字节与时间
计数完成，没有依赖 op 级汇总。

## 7. 最终推荐配置（本机长期有效）

- 通用/短 ctx（<=8K）：`-ncmoe 41 --moe-cache-mib 6144 -ub 2048 -t 5 -fa on` -> pp512 383 / tg32 26.2。
- 长上下文（ctx <= 32K，10k-20k prompt）：`-ncmoe 41 --moe-cache-mib 5120 -ub 2048 -c 32768` -> 411/347 t/s，21/20 t/s。
- 真 256K ctx：`-ncmoe 41 -c 262144 -ub 2048`（无 cache）-> 372/321 t/s，14 t/s 左右。
- 不要用 `-ncmoe 27` 静态配置跑长上下文（32K 就溢出）。
- 剩余理论空间：只有 GPU 计算（79% 中的 MoE GEMM 效率）与"GPU 直读 host 权重"（被 AMD 驱动堵死）；
  上传侧已贴 13.9 GB/s 天花板，调度层无可再挖。

原始日志：`bench/longctx/p1-*.{out,err}`、`v-*.out`，`p2-10k.err`（空，崩溃无输出）。
