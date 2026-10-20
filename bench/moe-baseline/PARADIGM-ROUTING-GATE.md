# T3 可行性门禁：跨 token routing 可预测性 —— 门禁失败，STOP（2026-10-20，task-25）

> 分支 `moe-gpu-cpu-hybrid` @ `3e3e7bbcb`。**结论：硬可行性门禁未通过，按任务要求 STOP，未实现特化 decode runtime。**
> 关键数字：跨 token 的 top-8 预测在 M=8/16/32 时 recall 只有 33.6-55.6%；30 个部分驻留层里
> **每个 token 至少一层不命中（token 级 fallback 率 100%）**；把预测超集常驻需要的上传量是
> 当前方案的 5-10 倍，PCIe 上界只有 4.4-15.2 t/s，全部低于现有 26.6 t/s。

## 1. 测量方法

- 在 `src/llama-context.cpp` 临时加 env 门控诊断 `LLAMA_ROUTING_DIAG=<file>`（已回退，见第 6 节代码片段）：
  graph build 时把 `ffn_moe_probs-<il>` 标记为 output 并在 `process_ubatch` 计算完成后读出，
  每 (层, token) 记录 probs 的 top-(8+32) ids（二进制追加）。
- 该架构的 actual top-8 就是同一 probs 的 top-8（无 bias/group mask），所以一对数据同时给出
  "预测"（token t 的 probs）和"真实"（token t+1 的 top-8）。
- 数据集：514-token prompt（`bench/moe-baseline/p6-prompt.txt`）；`bench/longctx/prompt10k.txt`
  （10428 tokens）。配对数量：513 / 10432 per layer，40 个 MoE 层。
- 分析脚本：`bench/moe-baseline/routing_gate.py`（提交）。二进制 dump（3.2MB / 65MB）未提交。
- 字节假设：每个 expert 三元组（gate+up+down）≈1.945 MiB（5 层 down 为 q4_K，略小，误差 ~2%）。
  H2D 天花板 13.9 GB/s（实测值）。

## 2. 门禁结果（每 token，30-40 层求和）

| M | 超集/层 | recall（514 prompt / 10k） | layer miss | **token fallback** | 超集常驻字节/token | 超集上界 | miss 字节上界* | 增量上界** |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 33.6% / 38.2% | 61.8-66.4% | **100%** | 1273 MB | **10.9 t/s** | 32.9-35.4 t/s | 15.2-16.0 t/s |
| 16 | 24 | 41.3% / 45.7% | 54.3-58.7% | **100%** | 1909 MB | **7.3 t/s** | 37.2-40.2 t/s | 11.0-11.4 t/s |
| 32 | 40 | 51.9% / 55.6% | 44.4-48.1% | **100%** | 3182 MB | **4.4 t/s** | 45.4-49.2 t/s | 7.7-7.9 t/s |

\* Lead 给的 `13.9 GB/s / miss_bytes` 公式（把预取流量当成免费），仅供参考。
\** 增量 = |S_t \ S_{t-1}|，即"常驻上一 token 超集"时的真实新增上传量。

## 3. 为什么门禁失败（两条独立理由，任何一条都足以否决）

1. **token 级 fallback 率 = 100%**：30 个部分驻留层，只要一层有 1 个 expert 不在预测超集里，
   整个 token 就要回退到逐层路径（正确性不允许用错 expert 计算）。要求 token 级命中率 >90%
   需要每层 recall > 99.7%；实测即使 M=32（40 个候选 = 15.6% 的专家）每层 recall 也只有 55.6%。
   所以特化路径会**永远走 fallback**，只剩额外开销。
2. **字节上界全部低于现状**：超集常驻设计每 token 需要 1.27-3.18 GB 的上传（上界 4.4-10.9 t/s）；
   即使只上传增量也有 0.87-1.76 GB（上界 7.7-16.0 t/s）。而当前 LRU 方案每 token 只上传
   ~200 MB（上界 70-100 t/s），实际被往返开销限制在 26.6 t/s。**预测式预取把字节量放大了 5-10 倍，
   必然更慢。** 关键原因：LRU 已经常驻了每个 token 真正用到的专家（命中率 ~68%），
   预测超集反而强制上传大量用不到的专家。

## 4. 决定

按任务约定 STOP：不实现重排的 decode runtime。正确性/速度二者都无法达成（fallback 100% + 字节放大）。

## 5. 唯一可行的替代方向（建议作为后续任务）

问题不是"预测"而是"每层 CPU 往返"。正确做法是**把 LRU/slot 规划搬到 GPU**（不做预测，用真实 ids）：

- 在 Vulkan 里实现一个小的 slot-plan kernel：输入当前层的 ids、bank 的 slot 映射表与 LRU 链，
  输出 slot_map（+ 需要上传的 miss 列表）；host 只在 token 边界读一次 miss 列表（或读取
  "无 miss" 标志），然后一次性把 miss 上传。
- 这样 token 图里不再有 30 次 per-layer callback、没有 per-layer drain；字节量与现在相同（~200MB/token）。
- 预期：token 时间 ≈ GPU 数学 12 ms + 1-2 次提交/回读 ≈ 15-18 ms -> **TG 55-65 t/s**，
  且不需要预测、不需要 fallback（GPU 用真实 ids 规划，天然精确）。
- 工作量：1-2 个 session（新 shader + llama-moe-cache 的 host 侧状态镜像与同步 + 逐层/整 token 校验）。
  风险中等：需要保证 GPU 侧 LRU 状态与 host 完全一致（可保留 host 侧影子状态做校验回退）。

## 6. 诊断代码（已回退，记录以便复现）

三处插入点（env `LLAMA_ROUTING_DIAG` 门控）：
1) `llama-context.cpp` 顶部：`static const char * llama_routing_diag_path()` + 两个 map<int32, ggml_tensor*>。
2) `graph_get_cb()` 的 lambda 里：name == "ffn_moe_probs" 时 `ggml_set_output(cur)` 并记录指针。
3) `process_ubatch()` 里 graph_compute 成功后：`ggml_backend_sched_synchronize`，按层读出 probs，
   对每个 token 做 partial_sort 取 top-(8+32) 写入二进制（格式：n_tok,n_lay, per layer
   il,n_expert,n_sel,n_keep,expert_bytes, then n_tok x n_keep int32）。

注意：不要把 `ffn_moe_topk` 也标记 output（会在堆上引发崩溃，实测 0xC0000374/0xC0000005）；
actual ids 用 probs 的 top-8 推导即可。

## 7. 测量环境备注

门禁数据是统计量，与机器负载无关，有效。门禁后在 16:30 复测 canonical（`-ncmoe 41
--moe-cache-mib 6144`）只有 pp512 139-176、tg32 14.2（正常应为 401/26.4）——当时另有进程/会话
占用机器（dGPU 显存仍报告 7378 MiB free、CPU 3.5%，原因未定位）。本轮未保留任何代码改动，
树仍等于已验证的 `3e3e7bbcb`；正式复测应在独占窗口重做。
