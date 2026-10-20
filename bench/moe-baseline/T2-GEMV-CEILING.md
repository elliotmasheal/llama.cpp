# T2: dense decode GEMV / MUL_MAT_VEC - measured ceiling (2026-10-20)

Base HEAD `30d341467`. No code change kept: every Vulkan-side GEMV knob measured neutral or
worse on end-to-end TG, and the premise numbers turned out to be inflated by the profiler.

## 1. The profiler distorts decode by 2.6x, so the "74% -> 90%" premise does not hold

`GGML_VK_PERF_LOGGER` instruments timestamps between graph nodes, and every instrumentation point
adds a wait, so decode is measured far slower than it runs:

| run | tg32 |
|---|---:|
| no logger | 26.82 / 26.83 |
| `GGML_VK_PERF_LOGGER=1` | 10.42 |
| `GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_CONCURRENT=1` | 11.65 |

The task-23 op table (MUL_MAT_VEC 407 calls x 30.7 us = 10.34 ms/token, "74% of the bandwidth
floor") was taken under that logger, so both the absolute time and the efficiency are distorted.
Real kernel times must come from isolated microbenchmarks (test-backend-ops perf, no logger).

## 2. Real dense GEMV: already ~97% of the DRAM floor

Microbenchmark times (`test-backend-ops perf -o MUL_MAT`, Vulkan1) times the per-token call
counts from the op shape histogram:

| shape | calls/token | us/call | MB/call | GB/s |
|---|---:|---:|---:|---:|
| q6_K m=248320 n=1 k=2048 (output proj) | 1.03 | 1920.8 | 417.4 | 217 |
| q4_K m=8192 n=1 k=2048 | 26.8 | 38.4 | 9.44 | 246* |
| q4_K m=2048 n=1 k=4096 | 32.0 | 24.3 | 4.72 | 194* |
| q4_K m=4096 n=1 k=2048 | 30.9 | 16.0 | 4.72 | 295* |
| q4_K m=512 n=1 k=2048 | 96.9 | 6.86 | 0.59 | 86* |
| q4_K m=32 n=1 k=2048 | 61.9 | 4.42 | 0.037 | latency bound |
| q6_K m=2048 n=1 k=512 | 20.6 | 7.94 | 0.86 | 108* |
| q4_K m=2048 n=1 k=512 | 20.6 | 6.33 | 0.59 | 93* |

\* Navi22 has a 32 MB infinity cache, so isolated tensors below that size benchmark above the
224 GB/s DRAM number. In the model each token streams 1.31 GiB of dense weights, which thrashes
the cache, so those shapes run at DRAM speed in situ.

Total dense weight traffic per token is 1.31 GiB (the output projection alone is 417 MB, 32%),
which is 6.0 ms at 224 GB/s. The microbench-weighted sum is 6.2 ms, i.e. dense GEMV already runs
at about 97% of the memory floor. There is no 2.7 ms to recover here.

## 3. MoE matvec: one real kernel-level gap, but it does not move decode

| shape | calls/token | us/call | MB/call | GB/s |
|---|---:|---:|---:|---:|
| MUL_MAT_ID_VEC q4_K m=512 n=8 k=2048 ne=1690 b=1 | 61.9 | 56.9 | 4.72 | 83 |
| MUL_MAT_ID_VEC q6_K m=2048 n=8 k=512 ne=1690 | 15.5 | 31.3 | 0.86 | 27 |
| MUL_MAT_ID_VEC q4_K m=2048 n=8 k=512 ne=1690 | 15.5 | 24.5 | 0.59 | 24 |

The gate/up shape is the largest single GPU item (61.9 calls x 56.9 us = 3.5 ms/token) and it
runs at 83 GB/s, while the dense path reads the same 4.72 MB in 16.0 us (295 GB/s). So the id
matvec kernel has a roughly 3.5x kernel-level gap. It is not workgroup-count limited: forcing
rows per workgroup to 2, 4 or 8 leaves the kernel time unchanged (56.9, 59.3, 57.5, 57.8 us).

## 4. End-to-end A/B of every Vulkan GEMV knob: neutral or worse

Interleaved llama-bench, r=2..3, cache 6144 config:

| knob | tg32 |
|---|---:|
| baseline | 26.23 - 26.96 (run noise +-0.7) |
| dense rows/WG 1 -> 2 (q4_K int path) | 26.38 |
| dense rows/WG 2 -> 4 (f32 path, q6_K) | 26.28 |
| id rows/WG 1 -> 2 / 4 / 8 | 26.38 / 26.11 / 26.15 |
| large workgroups (256 threads) | 21.98 (-17%) |
| force MMVQ everywhere | 25.97 (-3.5%) |
| disable MMVQ everywhere | 26.61 / 26.82 / 26.94 (all above baseline, +0.9%) |

Only disabling MMVQ at n=1 shows a consistent (3/3 interleaved) +0.9%, and it changes the
accumulation path, so it was not kept for a 0.9% device-specific numerics change.

Why nothing moves: decode spends about 37 ms/token, of which the per-layer CPU<->GPU chain and
expert upload take 23-30 ms (task-23 doc, L9/L10). The GPU math is ~12 ms and already sits at the
DRAM floor for the dense part, so a 20% kernel change moves total TG by 1-2% at most. The
21.98 t/s result for large workgroups confirms the GPU phase is serialized with the rest, not
overlapped: making the GPU much slower costs TG 1:1, but making the biggest GPU kernel 3.5x
faster would only pay off once the sync/upload chain is fixed.

## 5. Acceptance battery on the unchanged tree (HEAD 30d341467)

| case | target | measured |
|---|---:|---:|
| tg32 (cache 6144) | >= 28 | **26.82 / 26.83** (not reached) |
| pp512 | >= 395 | 397.82 +-4.27 |
| pp2048 ub 2048 | >= 575 | 578.22 +-2.40 |
| long-ctx 10k ctx 32768 prefill / decode | >= 405 / >= 20.6 | 411.2 / 20.1 |
| 256K ctx prefill / decode | - / >= 14.0 | 371.7 / 14.5 |

The 10k decode 20.1 vs 20.6 is run variance on the unchanged binary (earlier runs 20.5-20.8).

## Conclusion

TG >= 28 is not reachable through dense GEMV: the dense matvec is at ~97% of the DRAM floor and
the whole GPU math phase is a minority of the token time. The measured TG ceiling stays 26.8-27.1
t/s. The decode critical path is the per-layer dependency chain (30 layers x ids readback and
expert upload, 23-30 ms/token), which L9/L10 already showed cannot be overlapped or shortened by
backend-side waits.

The one concrete kernel-level lead that remains is the MUL_MAT_ID decode matvec gather path
(83 GB/s vs 295 GB/s for the same bytes on the dense path). It is worth a separate task: inspect
the id indexing in mul_mat_vecq.comp and compare against routing n=1 through the
MUL_MAT_ID_MUL/GEMM path, but expect the end-to-end effect to be small until the sync chain moves.
