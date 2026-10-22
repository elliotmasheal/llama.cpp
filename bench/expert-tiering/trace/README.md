# M0：路由轨迹采集（LLAMA_MOE_TRACE）

日期：2026-10-21。状态：已实现并冒烟通过。改动文件：`src/llama-moe-cache.cpp`（默认关闭，零行为变化）。

## 用途

为重排（层内热度、共激活分组）与淘汰策略（LRU/SLRU/ARC 离线重放）提供**真实使用**的路由数据。
现有数字全部来自 bench 语料，不足以支撑重排决策。

## 开启与采集

```powershell
$env:LLAMA_MOE_TRACE = 'E:\fastLlama\bench\expert-tiering\trace\live.bin'
build-vulkan\bin\llama-cli.exe -m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -fa on -t 5 `
    -ncmoe 41 --moe-cache-mib 6144 -p Hello -n 64 --temp 0 -st
```

记录格式（本机字节序，int32）：`layer, n, ids[n]`，每次 MoE cache 查找一条，按图顺序写入，层号回绕处即一个 token。
文件以追加方式打开，便于跨会话累积；关闭进程时日志会打印记录数。

## 分析

```powershell
python bench\expert-tiering\trace\read_trace.py bench\expert-tiering\trace\live.bin --top 8
```

输出：每条 token 的记录数、逐层 distinct 专家数、top-k 覆盖、Gini、层内共选专家对。

## 冒烟结果

- 命令：`-p Hello -n 16`，cache 6144，19 token，569 条记录，40 层。
- 轨迹文件 [qwen36_smoke.bin](qwen36_smoke.bin) 32,968 字节，`read_trace.py` 可解析。
- 样本过小（19 token），只能验证设施；层内 distinct 专家 70-107/256、Gini 0.28-0.54、top8 覆盖 21%-50%，**不能作为重排依据**。

## 已知限制

1. 只记录**有 host 专家、且不是整层驻留**的层；整层驻留层走静态视图、没有 ids 回读，因此不在轨迹里（冒烟中缺 0-6 层）。
2. prefill 走 `copy_experts` 大 batch 路径，不经过 `copy()`，因此**不记录 prefill**；轨迹是 decode 轨迹，正好对应放置决策。
3. 轨迹只含 ids、不含 router 权重；如需加权热度，后续在 `copy()` 里一并回读 `weights`。
4. 采集本身有开销（每个 cache 层一次二进制写），只用于数据收集，不要放进性能基准。

## 基线说明

本机当前电源方案为「平衡」，同一构建下 `-ncmoe 41 --moe-cache-mib 6144 -ub 2048 -p 512 -n 32 -r 3` 实测
pp512 **334.61 ± 1.89**、tg32 **21.56 ± 0.20**，低于 AGENTS 记录的 383/26.4（整机功耗状态差异）。
后续所有对比必须使用同一 session 的基线，不要直接用 AGENTS 数字做前后对比。
