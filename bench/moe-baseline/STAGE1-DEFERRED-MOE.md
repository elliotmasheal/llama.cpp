# STAGE 1: deferred 上传接入 MoE decode（2026-10-20，task-28）

> 分支 `moe-gpu-cpu-hybrid` @ `70d205871`（STAGE 0）+ 本次 STAGE 1。**机制正确，但收益只有 TG +1.5%，
> 达不到 33 的验收；未作为默认路径（env 门控，默认关闭，原路径仍是默认 fallback）。**

## 1. 实现

- `llama_moe_cache::copy()`：多张 expert 张量的层里，**最后一张（down）** 的 miss 上传改用
  `ggml_backend_set_tensor_async_deferred()`（进入 STAGE 0 的 deferred 批次）；gate/up 仍走普通路径。
  记录 `deferred_bank/deferred_backend` 与计数（bytes/waits/missed）。
- `llama_moe_cache::wait_deferred(backend, graph)`：当某个 split 的图里有 `MUL_MAT_ID` 的 `src[0]`
  就是该 deferred bank（owner 或 view_src）时，`ggml_backend_wait_deferred(backend)` —— 每层恰好一次显式等待。
- `ggml_backend_sched_set_pre_split_callback()`（新 API）：每个 split 提交前回调；`llama_context` 注册
  `sched_pre_split`，转发到 cache 的 `wait_deferred`。
- 安全回退：若某个 deferred 批次在下一次 copy 前仍未被消费（`missed`），下一次 copy 做一次完整
  `ggml_backend_synchronize` 并计数 —— 保证永远不会用未到达的权重计算。
- 开关：`GGML_VK_DEFERRED_COPIES=1` + `LLAMA_MOE_DEFER_UPLOAD=1`（两者都开才生效；默认关闭）。
- 依赖规则：deferred 目的 = MoE bank 的 down 槽；该槽在本 token 内专用于该层 down 张量、不会被提前读；
  跨 token 由现有 per-token 同步保证；消费者（down MUL_MAT_ID）之前有一次显式 `wait_deferred`。

## 2. 正确性（全部通过）

| 测试 | 结果 |
|---|---|
| 514-token prompt，n=64，temp 0，on vs off | 生成文本逐字节一致（1501 chars） |
| prompt10k（prefill 10.4k + decode 48），on vs off | 一致（1437 chars） |
| 强制 miss（`--moe-cache-mib 512`，n=48），on vs off | 一致 |
| PPL（cache 6144，chunks 2） | 两侧均 `PPL = 4.0688 +/- 0.51413` |
| deferred 批次全部被消费（missed） | 0（514 配置 1795 waits；forced-miss 1960 waits） |
| 等待关系（`GGML_VK_LOG_WAITS=1`） | 每个 partial 层每 token 一次 `B=1`，其余 compute 不等 B |

## 3. 性能（干净窗口，17:00 复测 canonical 401.14 / 26.46 —— 污染已消失）

| 配置 | pp512 | tg32 |
|---|---:|---:|
| off（默认） | 401.84 / 401.31 / 400.91（401.35） | 26.62 / 26.44 / 26.44（**26.50**） |
| on（defer） | 398.50 / 398.32 / 398.52（398.45） | 26.94 / 26.98 / 26.79（**26.90**） |
| 变化 | -0.7% | **+1.5%** |

其它 gate（defer on）：pp2048 = **578.14**（>=575）；long-ctx 10k（cache 5120, ctx32768）prefill **412.1** / decode **22.2**（>=20.6 / >=405）；256K 10k prefill **371.8** / decode **14.5**（>=14.0）。全部通过，无回归。

## 4. 为什么只有 +1.5%（结构性原因）

- 只有 **最后一张 expert 张量（down）** 可以被 defer：它的消费者（down MUL_MAT_ID）前面还有 gate/up 两段
  计算；deferred 上传量实测 = **34% 的总上传字节**（4213 MiB / 约 12.4 GiB，514-prompt 运行）。
- 可隐藏的上限 = 每层 gate+up 计算 ≈ 0.1 ms，而 deferred 上传 ≈ 0.17 ms/层 -> 每层最多埋 0.07 ms
  ≈ 2.1 ms/token（理想）；实测只有 **0.6 ms/token**（+1.5%），其余被等待/提交开销吃掉。
- 剩下 66% 的上传（gate/up）**不能 defer**：它们的消费者就是紧邻的下一次 compute，defer 只会把等待
  平移而不隐藏。因此本机制的天花板 ≈ 27 t/s，远低于验收的 33。

## 5. 结论与建议

- STAGE 0 的后端机制（独立 transfer semaphore + 显式消费者 wait）**已经实现并验证**，默认关闭、零成本；
  STAGE 1 的 MoE 接入也是正确的（逐字节一致 + PPL 一致 + 0 missed），但只带来 +1.5% TG，
  **未达到 33 的验收**，因此保持 env 门控、默认关闭。
- 要继续提升 TG，必须让**gate/up 的上传也被隐藏**，而它们的消费者紧邻计算；唯一办法是把 upload
  与 compute 放到不同且可独立等待的引擎/批次上并重排计算（例如把 gate/up 的 GEMM 换成两部分：
  先算已上传的 expert、再补算后到的）——这已经超出"调度层"范围，且会改变 MUL_MAT_ID 的语义，
  风险高。当前这台 8 GiB 机器的 decode 上限仍按 26.6（默认路径）报告。
- 环境污染在 17:00 消失（canonical 401.14 / 26.46），后续测量应尽快在干净窗口完成。
