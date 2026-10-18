# GPU + CPU 协同 MoE（hybrid）实施计划

> 分支 `moe-gpu-cpu-hybrid`，基线 `p1-moe-cache-async` @ `c127e9fe8`。
> 独立工作副本：`E:\fastLlama\.worktrees\moe-hybrid`（沙箱只对工作区内新建目录可写，
> 且 `git worktree` / 本地 `git clone` 的 ref 写入与传输被沙箱阻断，故用整树复制；
> 最终用一条提权命令 `git fetch <worktree> moe-gpu-cpu-hybrid` 取回主仓库）。

## 1. 本轮实测（2026-10-18，`-dev Vulkan1 -ngl 99 -fa on`，TG-only）

### 1.1 `-ncmoe 41`（全部 41 层 MoE 在 CPU），`GGML_SCHED_TRACE=1`

| 线程 | CPU 计算 | 拷贝 | 拷贝次数 | GPU 提交 | splits |
|---:|---:|---:|---:|---:|---:|
| `-t 1` | 66.2 / 95.9 / 68.4 ms | 20.8-23.5 ms | 130 | 8.2-13.9 ms | 82 |
| `-t 16` | 61.5 / 57.0 / 61.7 ms | 24.8-25.5 ms | 130 | 10.5-11.2 ms | 82 |

- `cpy_fb = 122`：绝大多数跨后端拷贝走同步回退路径；`cpy_sync = 11.6-16.0 ms`。
- **线程数从 1 增到 16，CPU 计算只快约 1.2x** -> CPU MoE 没有随线程扩展。

### 1.2 CPU 侧 op 归因（`GGML_CPU_TRACE=1`，`-t 8`）

每层 CPU split：`total = 1.3-2.8 ms`，其中 `MUL_MAT_ID = 1.26-2.70 ms / 3 节点`，
`GLU = 0.07 ms`。即 CPU 时间几乎全部在 MoE 的 3 个 `MUL_MAT_ID`。

按字节折算：41 层 x 8 专家 x 1.77 MB = 582 MB，60 ms -> **9.7 GB/s**；
按算力折算约 **33 GFLOPS 聚合（约 4 GFLOPS/线程）** -> **算力/kernel 受限，不是带宽受限**。

### 1.3 结论

1. 现状 TG（`-ncmoe 27`，14.2 t/s，70 ms/token）的主成本是 **CPU MoE 约 47 ms**，
   其次是 **每层跨后端往返约 20 ms（91-130 次 x 0.19-0.22 ms）**，GPU 提交约 13 ms，
   GPU 实际计算仅 12-15 ms。
2. 所以"协同"的第一杠杆不是搬运，而是 **降低 CPU 需要算的专家比例** 和
   **压掉每层往返的调用开销**；否则 CPU 侧永远吃满 token 预算。
3. 现有 `--moe-cache-mib` 的负优化来自每层回读 ids + 全设备同步（41 次/token），
   不是命中率本身：历史实测 **22% 覆盖率就有 65.4% 命中率**（路由分布偏斜明显），
   专家级驻留值得做。

## 2. 方案（按验证顺序）

### Phase A：消除每层跨后端往返开销

- 目标：`-ncmoe 27` 下把 copy+submit 的 ~33 ms 压到 < 8 ms。
- 手段：D2H/H2D 走 pinned staging + event，不再 `ggml_backend_synchronize` 全队列；
  合并同层相邻小拷贝；避免每次拷贝重新建 staging。
- 验收：`sched_trace` 的 `copy` 与 `cpy_sync` 显著下降；TG 不退化为前提。

### Phase B：专家级驻留 + CPU 算 miss（本次主功能）

- GPU 侧：所有层共享专家 bank（约 6 GB），**静态驻留表** `resident[layer][expert] -> slot`
  由 host 维护，miss 映射到 zero slot；图构建阶段**不回读 ids、不等上传**。
- CPU 侧：按需算 miss 专家（用 host mmap 原权重），与 GPU 命中的专家**并行**；
  每层一次 rendezvous（ids/激活 D2H、部分和 H2D）。
- 正确性：精确（不投机、不预测）；zero slot 参与 `MUL_MAT_ID` 但权重为 0。
- 后台上传：只往"当前 token slot_map 未引用"的槽位晋升，无 WAR 风险、无 deadline。
- 验收：`-ncmoe 41 --moe-cache-mib ~6000` 的 TG 从 11.2 t/s 提升；命中率 >= 65% 时不劣化。

### Phase C：CPU 侧效率与预算分配

- CPU `MUL_MAT_ID` GEMV 效率（repack 路径 / 线程绑定 / NUMA）；
- 按层路由偏斜分配 bank 槽位（当前按 host 字节均分）。

## 2.1 Phase B 实现规格（定稿）

### 数据流（每层，仅小 batch；PP 走原路径）

1. host 在 token 边界维护**驻留表** `resident[group][layer][expert] -> slot`（-1 = 未驻留），
   不随 token 内 ids 变化；slot 总数 = bank 槽位数，最后一个槽是**全零槽**。
2. 图内：`slots = GET_ROWS(resident_map_layer, ids)`，未驻留 -> 零槽，GPU 分支照旧
   （`MUL_MAT_ID` 三连 + 加权 + 聚合），零槽贡献 0，**GPU 分支不需要 mask**。
3. host 回调（读 32B ids 后）为每层写两张 host 小表：`slot_map` 不变式已在 1 完成，
   另写 `miss_mask`（I32[n_expert_used, n_tokens]）；不做任何专家上传。
4. CPU 分支：用**原始 host 专家张量** + `ids` + `miss_mask`，通过新的
   `GGML_OP_MUL_MAT_ID_MASK` 三连跑 gate/up/down（内核按 mask 压缩行表，**不读也不算**
   被 mask 的行），再加权/聚合，输出 `[n_embd, n_tokens]` 拷回 GPU 与 GPU 分支相加。
5. 精确正确性：不预测、不投机；miss 由 CPU 用原权重精确计算。
6. 后台上传只写**当前 token 驻留表未引用**的槽位；token 开始时等待传输 fence，
   然后把已完成的上传并入驻留表（对下一 token 生效），淘汰只在 token 边界发生。

### 新算子

- `GGML_OP_MUL_MAT_ID_MASK`：`(as, b, ids, mask) -> dst`，mask 为 host I32，shape 同 ids；
  仅 CPU 实现，其它 backend `supports_op=false`；内核复用现有 `mul_mat_id` 的行表构建，
  跳过 mask==0 的 (token, slot) 对；dst 先清零，避免未选中槽位残留。
- `ggml.c/h` 增加构造器与 op 名称；`ggml-cpu` 增加 case；图里通过 `ggml_backend_sched`
  自动落到 CPU split。

### 代码改动点

| 文件 | 改动 |
|---|---|
| `src/llama-moe-cache.{h,cpp}` | 驻留表/LRU → token 边界更新；`copy()` 改为写 slot_map + miss_mask、无关键路径上传；零槽；后台上传队列 |
| `src/llama-graph.{h,cpp}` | `build_ffn`/`build_moe_ffn` 增加 hybrid 分支（CPU masked 三连 + add），仅小 batch |
| `ggml/include/ggml.h`, `ggml/src/ggml.c` | 新 op 枚举/构造器/名称 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | masked 内核 |
| `src/llama-context.{h,cpp}` | 开关接线（`--moe-hybrid` 或 env），与 `--moe-cache-mib` 复用 |

### 验收

- 正确性：`--temp 0` 输出与 `-ncmoe 27` 基线逐字符一致（同一 prompt/seed）；`test-backend-ops` 新 op CPU 通过。
- 性能：`-ncmoe 41 --moe-cache-mib 6000 --moe-hybrid` 的 TG > `-ncmoe 27` 基线 14 t/s；
  期望 25-35 t/s（命中率 65-75%）；PP 不退化（大 batch 走原路径）。

## 3. 环境阻塞（已解决）

本轮已切换到完全权限：`cmake` 配置与 759 个目标编译通过，worktree 二进制可用。
（受限模式下 `cmake -B` 的探针编译会因沙箱禁止孙进程管道而挂起。）

本会话的受限 shell **无法编译**：`cmake -B` 需要探针编译并捕获编译器输出，
被沙箱的"孙进程管道"限制挂起（`build.ninja`/`CMakeCache.txt` 均未生成，ninja 空转）。
需要以下之一：

1. 会话切到完全权限（推荐，后续所有增量编译可直接跑）；
2. 每次构建由我发起一条提权命令，人工批准（每次 1 次批准）。
