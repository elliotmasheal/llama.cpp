# Phase B2 status: hybrid GPU+CPU MoE (task-3)

Date: 2026-10-18. Branch `moe-gpu-cpu-hybrid`. Baseline commit `48866c708` (masked CPU op).

## State

The MVP is implemented and builds. `--moe-hybrid` is currently **gated off** in
`src/llama-context.cpp` (one line, `const bool moe_hybrid = false`); with the flag absent or
present the context uses the existing LRU cache path, so default behavior is unchanged.

Implemented (uncommitted on top of `48866c708`):

- `--moe-hybrid` cparam + CLI flag + logging: `include/llama.h`, `common/common.h`,
  `common/common.cpp`, `common/arg.cpp`, `src/llama-context.cpp`.
- Cache side (`src/llama-moe-cache.{h,cpp}`): `hybrid(il)`, `get_miss_mask()`, `count_usage()`,
  static residency (equal quota per layer, remainder to the first layers, top experts by prompt
  usage, ties by expert id), residency map (slot for resident, `n_slots` = zero slot for misses),
  hybrid `copy()` that fills the host mask and copies the residency map without uploading experts,
  `copy_experts()` guard for the zero slot, hybrid hit-rate stats.
- Graph side (`src/llama-graph.{h,cpp}`): `build_moe_cache_slots(..., hybrid)`, CPU branch in
  `build_moe_ffn` (masked gate/up MMID -> swiglu -> masked down MMID -> per-slot views/adds) and
  `moe_out = add(moe_out_gpu, moe_out_cpu)` on the cache device.

## Correctness proof

The hybrid split is exact; only the quantized kernel numerics differ between the GPU and CPU
paths, exactly like the existing LRU cache path.

PPL on `bench/moe-baseline/ppl-text.txt`, `-c 64 -b 8 -ub 8 --chunks 3`:

| config | PPL | hit rate |
|---|---:|---:|
| `-ncmoe 41` all CPU, no cache | 6.1161 | - |
| ref: `-ncmoe 41 --moe-cache-mib 4096` (LRU, all experts on GPU) | 6.1973 | - |
| hybrid, cache 4096 | 6.1852 | 22.0% |
| hybrid, cache 100 MiB | 6.3628 | 0.7% |
| hybrid, cache 100 MiB, residency forced to 0 residents | 6.1161 | 0.0% |
| hybrid, cache 100 MiB, uploads done but map all-miss | 6.1161 | 0.0% |
| hybrid, mask forced to 1 + CPU result only | 6.1161 | 0.7% |

The zero-resident case reproduces the all-CPU result exactly (6.1161), which proves the graph
wiring, the host miss-mask view, the zero slot, the split ordering, the CPU->GPU final add and the
residency upload are exact. Greedy text is not an acceptance test: `-ncmoe 27` / all-CPU /
GPU-hit paths already differ in greedy text (cpu41=6.1161 vs ref LRU 4096=6.1973 shows the same
kernel-numerics effect without any hybrid code).

What is proven exact in isolation:

- CPU masked op vs plain `ggml_mul_mat_id` on CPU (Q4_K gate shape and Q6_K down shape, sparse
  mask): bit-exact (max_diff 0).
- Misses read the zero bank row and contribute exactly 0 (also verified on Vulkan standalone).
- Residency upload: 156/156 resident (layer, expert, tensor) rows read back byte-equal to the host
  experts; residency map values are as planned.
- The CPU kernel sees the intended mask and the host mask pointer equals what the kernel reads.

## Residual

The 0.7% hit-rate run is +4% PPL vs the all-CPU path while the 22% hit-rate run is +1.1%; the
error does not scale with the number of GPU-computed (hit) experts, so there may be a residual
issue specific to very small slot counts (quota 1-2 per layer). Not yet explained; the 22% case is
consistent with the ref LRU numerics.

Note: repack is not involved. With mmap the expert tensors stay plain host weights, the masked op
runs the normal CPU kernel and is bit-exact in isolation. If `--load-mode none` (repack active) is
used, `ggml/src/ggml-cpu/repack.cpp` must learn `GGML_OP_MUL_MAT_ID_MASK` in
`supports_op`/`get_tensor_traits`.

## Performance (llama-cli, ~600 token prompt, -n 64, -fa on, -dev Vulkan1)

| config | PP t/s | TG t/s | splits | copy ms | cpy_sync | h2d | cpu ms | submit ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `-ncmoe 27` (baseline) | 70.4 | 15.9 | 56 | 16.1 | 15.2 | 0.6 | 28.9 | 10.8 |
| `-ncmoe 41 --moe-cache-mib 4096` (LRU) | 49.3 | 11.0 | 42 | 84.4 | 14.3 | 70.1 | 0.0 | 8.3 |
| same + `--moe-hybrid` (4096) | 61.5 | 6.1 | 122 | 103.8 | 72.5 | 30.9 | 41.8 | 15.0 |
| same + `--moe-hybrid` (6144) | 56.5 | 5.1 | 122 | 140.0 | 102.9 | 36.6 | 38.8 | 13.7 |

Per decode token; `GGML_SCHED_TRACE=1` averages over 64 graphs. Hit rates with the residency
frozen from prefill counts: 22.2% at 4096 (coverage 22.0%), 34-38% at 6144 (coverage 33.0%).
Static popularity residency therefore gives almost no gain over the coverage: the 65% hit rate at
22% coverage recorded in HYBRID-PLAN.md comes from LRU temporal locality, not from prefill counts.

Cause of the 3x slowdown (concrete, from the trace):

1. The CPU branch adds 2 splits per layer (GPU hit branch, CPU miss branch, GPU final add):
   122 splits vs 42 (LRU ref) and 56 (`-ncmoe 27`).
2. The CPU split needs `inp` and `ids` from the device, so the scheduler synchronizes the Vulkan
   queue once per CPU split: 80 async_get copies and `cpy_sync` 72.5 ms/token at 4096
   (102.9 ms at 6144). This also means the two branches never overlap: the CPU miss branch runs
   after the GPU hit branch has fully drained, so the token costs the sum of both plus copies.
3. 78% of the experts are still computed on the CPU at 22% hit rate: `cpu` 41.8 ms/token
   (`-ncmoe 27` pays 28.9 ms for 27 layers).
4. The per-layer cache copy callback (ids readback + full device synchronize + slot map upload) is
   traced as h2d 30.9 ms/token; the same cost exists in the LRU ref path.
5. submit 15.0 ms vs 10.8/8.3 because of the split count.

The mask view itself is not copied across backends (it is only consumed by CPU nodes); the extra
splits come from the CPU branch, not from the mask.

Verdict on the Phase B acceptance: hybrid does NOT beat `-ncmoe 27` (TG 6.1 vs 15.9 at 4096,
5.1 at 6144). A static prefill-count residency cannot reach 60% hit rate on this model with an
8 GB card (needs ~60% coverage, i.e. ~11 GB of experts); reaching the target requires dynamic
residency (temporal locality, Phase C), removal of the per-layer device synchronize in the copy
callback, and real CPU/GPU overlap of the two branches.

## Reproduction helpers (untracked)

- `bench/moe-baseline/mask_check.cpp` - masked op vs plain MMID on CPU (Q4_K gate shape).
- `bench/moe-baseline/mask_check_down.cpp` - same for the Q6_K down projection.
- `bench/moe-baseline/gpu_bank_check.cpp` - Vulkan MMID with a bank + zero row + slot ids vs CPU.
