# DDD Slice 0 基线 (2026-10-29)

状态: baseline snapshot; 本次没有重跑性能矩阵, 没有改运行时行为
范围: qwen35moe / Vulkan + CPU / VRAM-RAM-SSD
配套: [ddd-refactor-plan.md](ddd-refactor-plan.md), [common/residency.h](../../common/residency.h), [tests/test-residency-plan.cpp](../../tests/test-residency-plan.cpp)

> 本文件只做基线与盘点, 不是 bench 结论文档。性能数字仍以 `bench/RESULTS.md` 为准。

## 1. 性能基线

目标口径: 256K ctx + 10K prompt, IQ4_XS, 8176 MiB 档, 2026-10-29。

| 指标 | 读数 | 来源 |
|---|---:|---|
| PP | 567.4 t/s | `bench/RESULTS.md` 第 0 节 |
| TG | 20.66 t/s | 同上 |
| 专家命中 | 68.04% | 同上 |
| bank | 2376.25 MiB | 同上 |
| uploaded | 9727.62 MiB | 同上 |
| KV 工作集 | 约 320 MiB | `bench/RESULTS.md` 第 4 节 |
| SSD 随机 | 0.81 GB/s | `bench/RESULTS.md` 第 1 节 |

本次没有重跑矩阵, 以上是快照。任何 call-site 改动前先跑:

```
bash bench/tools/run-matrix.sh -T 8176,6144,4096 -r 2
```

## 2. 四处 abort 盘点

`AGENTS.md` / `bench/THREE-TIER.md` 里的 `:1841` / `:2157` / `:2189` 是旧行号; 当前实际位置如下。

| 实际位置 | 函数 | 触发条件 | Slice 1 决策 | 调用方动作 |
|---|---|---|---|---|
| `src/llama-moe-cache.cpp:129` | `moe_cache_slru::victim` | 当前批次选中的专家覆盖了所有可用 slot, 找不到可淘汰 victim | `on_no_slru_victim` -> `DROP_ROLLING` | `plan` 返回 false; 关闭 rolling, 保留静态 bank, miss 走 host/CPU |
| `src/llama-moe-cache.cpp:1843` | `hybrid_combine` | hybrid worker 失败 | `on_hybrid_worker_failed` -> `DISABLE_HYBRID` + `fail_current_decode` | 当前 decode 返回失败; 下一次禁用 hybrid, 走图内 CPU 分支 |
| `src/llama-moe-cache.cpp:2184` | rolling 的 `copy_experts` | 本层选中的不同专家数超过 slot 数 | `on_cache_too_small(true)` -> `DROP_ROLLING` | 关闭 rolling, 回退到非 rolling 路径 |
| `src/llama-moe-cache.cpp:2216` | 非 rolling 的 `copy_experts` | 同上, 但当前不在 rolling | `on_cache_too_small(false)` -> `DISABLE_CACHE` + `DISABLE_HYBRID` | 关闭 bank/hybrid, 专家全部走 host/CPU 分支 |

边界说明: `:1843` 是 worker 基础设施失败, 不是容量不足; 当前 token 的结果已经缺失, 所以策略允许 `fail_current_decode`。其余三处是容量/驻留条件, 必须退化并计数。

## 3. 跨层搬运与调用点

这些是 Slice 2 要收敛的路径, 本次只盘点:

- `src/llama-moe-cache.cpp:1334-1440` 预取 API 初始化、ring 规划。
- `src/llama-moe-cache.cpp:1371-1406` 按 `--store-exec` 路由到 store-io。
- `src/llama-context.cpp:2965` scheduler copy callback 的专家拷贝。
- `common/common.cpp:1344-1478` store 初始化、declare、adopt、commit。
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:13686-13771` 五个 prefetch 原语。

目标: 不新增第三套跨层搬运; 只在现有 store-io / prefetch 原语上加窄接口。

## 4. Slice 1 草案

已落盘, 但未接运行时:

- `common/residency.h` / `common/residency.cpp`: 纯结构 + solver + degradation policy + counters; 没有加入 `common/CMakeLists.txt`, 不会被 `llama` / `llama-common` 链接。solver 只是测试 oracle, 不是第二套生产规划器; `common_moe_plan_get` 仍是生产口径的唯一所有者。
- `tests/test-residency-plan.cpp`: 只编译 `common/residency.cpp`, 不需要 GPU / 模型。
- `tests/CMakeLists.txt`: 注册 `test-residency-plan`。

已固定的不变量:

- 三个预算互不挪用; 任一 tier 超支不能被另一 tier 的空闲空间覆盖。
- `sum(alloc) <= budget`。
- 分配顺序 KV -> bank -> compute; compute 放不下时先牺牲 bank, 仍放不下才拒绝。
- 负数/超预算返回明确 status, 不 abort。
- 四种 abort 决策: `DROP_ROLLING`、`DROP_ROLLING`、`DISABLE_CACHE + DISABLE_HYBRID`、`DISABLE_HYBRID + fail_current_decode`。
- 退化计数: 每种 reason 一个 counter。

验证命令与结果:

```
cmake --build build-vulkan --target test-residency-plan -j 8
build-vulkan/bin/test-residency-plan.exe   # test-residency-plan: OK
ctest --test-dir build-vulkan -R test-residency-plan --output-on-failure  # 1/1 passed
```

注意: ctest 现为 35 项（34/35, `test-model-resolution` 环境性失败）; 本基线之外, 四处 abort 已在 0.17.10 就地退化。

## 5. 接运行时前的门禁

1. 重跑目标口径矩阵, 确认基线。
2. 四处 abort 逐个替换为 policy + counter; 每次只改一处, 单独跑矩阵。
3. 每处都要有 `status = ok` 的降级路径, 不允许用新开关绕过。
4. PP/TG 在 r=2 噪声内; 动数值路径再加 quality gate。
5. 更新 `bench/` 受影响文档与 DDD 模型状态; 不新增第五份 bench md。

## 6. 本次没有做什么

- 没有改 `src/llama-moe-cache.cpp` 的四处 abort。
- 没有改 `common/CMakeLists.txt` 或运行时库。
- 没有改 `fit` 的求解逻辑。
- 没有新增永久开关。
- 没有重跑 PP/TG 矩阵。
