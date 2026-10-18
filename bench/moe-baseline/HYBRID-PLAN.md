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

## 3. 环境阻塞（待解决）

本会话的受限 shell **无法编译**：`cmake -B` 需要探针编译并捕获编译器输出，
被沙箱的"孙进程管道"限制挂起（`build.ninja`/`CMakeCache.txt` 均未生成，ninja 空转）。
需要以下之一：

1. 会话切到完全权限（推荐，后续所有增量编译可直接跑）；
2. 每次构建由我发起一条提权命令，人工批准（每次 1 次批准）。
