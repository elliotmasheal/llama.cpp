# PP baseline: upload vs compute split for qwen35moe 35B A3B (pp2048 ub2048)

Measurement operator: teammate pp-measure. Session: 2026-10-21 23:07-23:22 (Asia/Shanghai).
Binary: E:\fastLlama\build-vulkan\bin (build a9d2cc8a7, 11750). No source changes, no build.
Model: F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf (qwen35moe, 41 blocks, 256 experts, top-8,
expert_ff 512, n_embd 2048; expert bytes 18.650 GiB, one expert-layer = 1.820 MiB average).
GPU: RX 6700S (Vulkan1), 8176 MiB, exclusive for every run (Get-Process llama* checked by
pp-tools/run_pp.ps1 before each start). All runs below are one session, one power state
(no power-plan change). Logs: bench/expert-tiering/pp-baseline/.

SESSION POWER STATE (important): this session runs ~25% below the AGENTS.md canonical
numbers (pp512 332 vs 383-387). The earlier trace README already recorded pp512 334.61 in
this same "balanced" state, so compare only within this session; the AGENTS pp2048 578-590
is not reproducible here (see "scaling" note in the summary).

## T1 baseline (r>=2, same session)

| # | test | command tail | t/s (as printed) | log |
|---|---|---|---|---|
| a | pp2048 ub2048 cache 6144 | `-ncmoe 41 --moe-cache-mib 6144 -p 2048 -n 0 -ub 2048 -r 3` | **439.36 +- 1.75** | t1a-retry-cache-pp2048-ub2048.log |
| b | pp512 ub2048 cache 6144 | `-p 512 -n 0 -ub 2048 -r 3` | **332.25 +- 1.95** | t1b-cache-pp512-ub2048.log |
| c | pp512 + tg32, cache 6144 | `-p 512 -n 32 -r 3` (ub 2048) | pp512 **329.28 +- 2.24**, tg32 **20.08 +- 0.37** | t1c-cache-pp512-tg32-ub2048.log |
| d | pp2048 ub2048 static control | `-ncmoe 27 -p 2048 -n 0 -ub 2048 -r 2` | **429.18 +- 0.22** | t1d-static27-pp2048-ub2048.log |

Common prefix for all: `-m F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -fa on -t 5`.

T1a FAILED on the first attempt: `terminate called after throwing an instance of
'vk::OutOfDeviceMemoryError' ... vk::Queue::submit: ErrorOutOfDeviceMemory`
(t1a-cache-pp2048-ub2048.log, exit -1073740791, 53.3 s). The identical command succeeded on
retry (t1a-retry). VRAM is therefore at the edge for pp2048 + cache 6144, not stable
(see T5). pp512/tg32/static are stable.

## T2 time split (cache 6144, pp2048 ub2048, -r 1, GGML_SCHED_TRACE=1 + GGML_MOE_COPY_TRACE=1
+ GGML_VK_MEMORY_LOGGER=1). Logs: t2-trace-pp2048.log / .log.err. Measured eval t/s 432.18.

Raw scheduler trace (measured eval, second line; first line is the warmup eval):

```
sched_trace: splits=90 sync=0.000ms copy=4670.234ms (mb=56.078 n=10 hw=88 async=2 asyn_get=0
  fb=0 cpy_sync=4583.278ms d2h=0.000ms/0 h2d=86.750ms/98) cpu=2.565ms gpu_submit=47.778ms
```

- splits per eval: 90; host-weight expert-tensor copies: 88 (29 cached MoE layers x 3 tensors + 1)
- copy phase 4670 ms of the 4739 ms eval, of which **cpy_sync = 4583 ms** is the wait for the
  previous split's GPU work. That is the serialization: the CPU cannot upload split N+1 until
  the GPU has finished split N, so the eval wall is ~GPU-busy time, not upload + compute.
- `h2d = 86.75 ms / 98 copies` is **CPU-side enqueue time only, not DMA time**: the host expert
  model buffer is a pinned (host-visible) Vulkan buffer, so
  ggml_vk_buffer_write_2d_async records a vkCmdCopyBuffer from pinned memory into the split
  command buffer (no staging memcpy). The DMA executes on the GPU queue inside the 4583 ms wait.
- d2h = 0: for ub 2048 the routing ids are read back inside the copy callback (sync), not as a
  split input, so this counter does not show them; mb=56 MiB / 10 copies are activation copies.

Upload bytes per eval (GGML_MOE_COPY_TRACE, 87 lines per eval, 29 layers x 3 tensors):

| item | eval 1 (warmup) | eval 2 (measured) |
|---|---|---|
| host -> device expert upload | 9004.9 MiB | **8795.7 MiB = 8.59 GiB = 9.22 GB** |
| device bank -> device (D2D) | 1104.9 MiB | **1055.5 MiB = 1.03 GiB** |
| total expert traffic | 10109.8 MiB | 9851.2 MiB |

Derived time split for the measured eval (wall 4739 ms = 2048 / 432.18):

| component | value | source |
|---|---|---|
| eval wall | 4739 ms | llama-bench t/s |
| GPU busy (sum of split waits) | 4583 ms | sched_trace cpy_sync |
| upload DMA at 13.9 GB/s (probe ceiling) | 663 ms | 9.22 GB / 13.9 |
| upload DMA at ~9.5 GB/s (gather, est.) | 970 ms | see T4 |
| bank D2D 1.03 GiB at 74.6 GB/s | 15 ms | probe D2D full bulk |
| **compute (busy - upload - D2D)** | **3600-3900 ms** | derived |
| upload share of eval | 14-20% | derived |

Note on the static control: pp2048 `-ncmoe 27` (12.6 GiB contiguous upload) = 429.18 t/s,
vs cache 439.36 t/s (9.2 GiB gather + 1.0 GiB D2D). Nearly equal wall with 40% more bytes
means contiguous whole-tensor upload runs ~14 GB/s while the gather path runs ~9-10 GB/s.
Additional static points (r2): `-ncmoe 41` 394.77 +- 4.82, `-ncmoe 13` 261.67 +- 1.99
(the latter over-commits VRAM: 27 layers of experts on an 8 GiB card; not a clean point).

GGML_VK_PERF_LOGGER=1 is NOT usable for this config: it crashes with
`GGML_ASSERT(ctx->compute_ctx.expired()) failed` (t2-perflog-pp2048.log.err) because the MoE
cache callback keeps a compute context alive. On the static `-ncmoe 27` config it survives and
prints 164 per-split/per-group tables; the raw lines are in
t2-static27-perflog-pp2048.log.err, the two whole-eval aggregates are
`Total time: 1.24095e+06 us` and `1.24268e+06 us` (1.24 s of GPU node time per eval).
Interpretation: 1.24 s is a LOWER BOUND on compute (timestamps cover ggml nodes only, not the
H2D copies and not inter-kernel gaps); the perf-logger run also reported pp2048 325.68
(slower than 424 without the logger), so its wall time is not comparable. Cross-check: the
sched-trace decomposition gives 3.6-3.9 s of non-upload GPU busy, so ~2.4 s of GPU busy is
not attributed to ggml node timestamps. Treat perf-logger "Total time" as unreliable here;
the sched-trace split is the defensible number.

## T3 expert coverage at a given ubatch (prefill path)

Tool: bench/expert-tiering/pp-tools/trace_coverage.py (two modes).
LLAMA_MOE_TRACE is supported and takes a file path, but **prefill is not traced**: the
traced `copy()` path is gated by `max_batch = 32` (src/llama-moe-cache.cpp), so
`-p 2048 -ub 2048` writes only 2 records (layer 39, 8 ids = one 1-token graph edge,
moe-trace-pp2048.bin, 80 bytes). Coverage for prefill is therefore derived from
GGML_MOE_COPY_TRACE, where `(host + bank) / bytes_per_expert` is exactly the number of
distinct experts the copy callback uploaded/copied for that layer (sched_copy_experts unions
the ids of the whole ubatch). Expert sizes: gate/up Q4_K = 589824 B, down Q6_K = 860160 B or
Q4_K = 589824 B (mixed per layer). The 5 whole-layer-resident layers emit no line and have
coverage 1.0 by construction.

ub2048 (t2-trace-pp2048.log.err, measured eval 2), 29 traced layers:

| item | value |
|---|---|
| distinct experts per traced layer | min 144, mean 188, max 237 |
| coverage = distinct / 256 | **min 0.562, mean 0.734, max 0.926** |
| gather bytes (29 layers) | 9849 MiB |
| whole-layer bytes (same 29 layers) | 13452 MiB (432 or 498 MiB per layer exact; 466 MiB at 1.820 avg) |
| whole-layer / gather | **1.37x** (1.26x by eval 1) |

ub512 (t3-cache-pp512-ub512-copytrace.log.err, measured eval 2), 29 traced layers:

| item | value |
|---|---|
| coverage | **min 0.434, mean 0.601, max 0.809** |
| gather bytes | 8067 MiB |
| whole-layer bytes | 13452 MiB |
| whole-layer / gather | **1.67x** |

Per-layer tables are in the log lines and can be regenerated with:
`python bench\expert-tiering\pp-tools\trace_coverage.py --moe-copy <log.err>`.

## T4 large contiguous H2D vs per-expert chunks

An existing, suitable probe already measures exactly this:
bench/h2d-bw/h2d-bw.exe (raw Vulkan, no ggml/llama dependency, BUILD.txt has the one-line
g++ command). It is a standalone probe, so nothing was built or modified for T4. Measured
output (prior session, bench/h2d-bw/T1-h2d-ceiling.out, 512 MiB per test, decimal GB/s):

| shape | path | GB/s |
|---|---|---|
| 1.5 MiB chunks, serial, 1 in flight | wc pinned -> device | 7.09 - 7.70 |
| 1.5 MiB chunks, 8 in flight (pipelined) | wc pinned -> device | 11.56 - 11.78 |
| 4 MiB chunks, serial / 8 in flight | wc pinned -> device | 10.32 / 12.87 |
| 16 MiB chunks, serial / 8 in flight | wc pinned -> device | 12.73 / 13.83 |
| 64 MiB chunks, 8 in flight | wc pinned -> device | 14.05 - 14.14 |
| full 512 MiB, one bulk copy | wc pinned -> device | **14.15 - 14.20** |
| D2D full bulk | device -> device | 74.63 (149.3 GB/s r+w) |

Answers: (i) per-expert gather at 1.82 MiB per chunk costs ~7.1-7.7 GB/s serialized, up to
~11.6 GB/s with 8 copies in flight; (ii) a 466 MiB whole-layer contiguous copy is
~14.2 GB/s (between the 64 MiB pipelined 14.05 and full-bulk 14.19 rows). Missing: the probe
was not re-run in this session and has no exact 1.82 MiB / 466 MiB rows; T2's engine-side
deduction (~9-10 GB/s effective for the current gather, which mixes 1-30 MiB runs) is the
proxy for what the runtime actually achieves.

## T5 VRAM headroom with cache 6144

Source: GGML_VK_MEMORY_LOGGER=1 run t2-trace-pp2048.log.err (65 alloc/free events).
- Peak "Total device" reported by the Vulkan allocator: **9.71 GiB = 9944 MiB**, against
  8176 MiB of VRAM. The driver/WDDM overcommits (system-RAM eviction), which is why the
  first T1a run died in vk::Queue::submit with ErrorOutOfDeviceMemory and the retry passed.
- Model device buffers: 1023.57 + 279.33 = 1302.9 MiB.
- MoE cache bank allocations: 4 x 951.19 MiB + 1.35 GiB + 458.08 MiB + 40 + 62.81 MiB
  ~= 5.5 GiB (the `--moe-cache-mib 6144` budget, split to respect the 2 GiB VkBuffer cap).
- pp2048-specific compute/logits buffer: 1.89 GiB (pp512 does not need it, which is why pp2048
  is the config that OOMs).
- Pinned host expert buffers: peak 18.94 GiB (system RAM, not VRAM).
Headroom at pp2048 + 6144 = effectively zero (negative before eviction). llama-bench does not
print model/context buffer sizes; the numbers above come from the Vulkan memory logger and
from llama-completion's log (t5-init.log.err) for the same flags.

## 10-line summary

1. Baseline (this session): pp2048 ub2048 cache 6144 = 439.36 +- 1.75 t/s; pp512 = 332.25 +- 1.95;
   tg32 = 20.08 +- 0.37; static -ncmoe 27 pp2048 = 429.18 +- 0.22.
2. Upload per pp2048 eval = 8795.7 MiB host->device (8.59 GiB / 9.22 GB) + 1055.5 MiB bank D2D.
3. Upload time = 663 ms at the 13.9 GB/s ceiling, ~970 ms at the ~9.5 GB/s gather rate the
   runtime actually gets; bank D2D = 15 ms. That is 14-20% of the 4739 ms eval.
4. GPU busy per eval = 4583 ms (sched_trace cpy_sync); splits = 90; expert tensor copies = 88.
5. Compute = 3600-3900 ms, i.e. 2048/compute = 525-569 t/s ceiling if upload were fully hidden.
6. Whole-layer vs gather bytes at ub2048 = 1.37x more (13452 vs 9849 MiB over 29 layers);
   at ub512 = 1.67x more.
7. Coverage at ub2048 = min 0.562 / mean 0.734 / max 0.926 per traced layer; 5 layers are
   already whole-layer resident (no upload at all).
8. Coverage at ub512 = min 0.434 / mean 0.601 / max 0.809.
9. VRAM: peak reported device allocation 9.71 GiB vs 8176 MiB; zero headroom, one OOM observed.
10. Implied pp2048 ceiling: 2048 / 3.6-3.9 s = 525-569 t/s (this power state). At 13.9 GB/s the
    same 9.22 GB upload costs 663 ms, so full overlap alone gives that 525-569 t/s; scaling to
    the documented 578-590 state (x1.33) gives ~700-756 t/s - still short of 780, so 780 needs
    overlap AND ~10% more compute throughput.
