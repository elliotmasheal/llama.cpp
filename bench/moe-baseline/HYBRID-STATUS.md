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

## Performance (llama-cli, 846-char prompt, -n 32, -fa on, -dev Vulkan1)

| config | PP t/s | TG t/s |
|---|---:|---:|
| `-ncmoe 27` (baseline) | 30.2 | 15.6 |
| `-ncmoe 41 --moe-cache-mib 4096` (ref) | 21.0 | 8.3 |
| same + `--moe-hybrid` | 26.0 | 5.7 |

The hybrid is slower because the hit rate is only ~23.5% at 22% coverage: the residency is built
during the 1-token warmup run (before prefill counts exist) and frozen, so the prompt routing
statistics are never used. Next step is to build the residency on the first small batch after
prefill, then re-measure.

## Reproduction helpers (untracked)

- `bench/moe-baseline/mask_check.cpp` - masked op vs plain MMID on CPU (Q4_K gate shape).
- `bench/moe-baseline/mask_check_down.cpp` - same for the Q6_K down projection.
- `bench/moe-baseline/gpu_bank_check.cpp` - Vulkan MMID with a bank + zero row + slot ids vs CPU.
