# P11: MUL_MAT_ID m/l tiles on the BK_STEP=4 variant - measured, not enabled (2026-10-22)

Base HEAD `a9d2cc8a7` (build 11750). Changed: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` only, no
shader regeneration. Closes item F1 of the AGENTS.md MoE audit: the id m/l tiles stay on the
BK_STEP=1 variant, and the full-BK4 path is kept behind `GGML_VK_ID_BK4` for hardware with more
LDS.

## 1. What was tried

P4 shipped dual id SPIR-V variants: the s tile runs BK_STEP=4, the m/l tiles run BK_STEP=1. The
audit hypothesis was that the m/l tiles should also run BK_STEP=4, because the PP prefill case
(batch 2048) selects the m tile:

    matmul_id_tile_selector (non-coopmat2), n = tokens per expert
    n <= 32              -> s tile  (BK_STEP=4 already)
    32 < n <= 64         -> m tile  (BK_STEP=1, used at ub 2048 with 256 experts top-8)
    n > 64               -> l tile  (BK_STEP=1, used at ub 4096+)

Implementation: `ggml_vk_id_bk4_all()` selects the bk4 blob for every id tile, and the id m/l
shmem checks pass `bk_step = 4`. Default is off after measuring; `GGML_VK_ID_BK4=1` opts in.

## 2. Measurement (same binary, env toggle, interleaved)

`llama-bench -m Ornith-1.5-35B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -ncmoe 27 -fa on -t 5
-p 2048 -n 0 -ub 2048 -r 3`. GPU idle, no other llama process.

| run | id m/l BK_STEP=1 | id m/l BK_STEP=4 |
|---|---:|---:|
| round 1 | 565.55 +-0.21 | 554.65 +-2.30 |
| round 2 | 563.76 +-1.29 | 558.62 +-0.33 |
| round 3 (default vs opt-in) | 565.03 +-1.30 | 558.71 +-2.39 |
| round 4 (default) | 565.83 +-0.23 | - |
| **mean** | **565.04** | **557.33** |

pp2048 loses **1.36%** with BK4 on the m/l tiles. The direction repeats in all four interleaved
comparisons, the spread inside a config is smaller than the gap.

pp512 is unaffected by construction: at batch 512 the tokens per expert count is 16, so the s
tile (already BK4 before this change) is selected. Measured control with the canonical
`-p 512 -n 32 -t 8 -r 3`: pp512 158.28 +-13.68 (default) vs 158.03 +-14.21 (BK4), i.e. inside the
run-to-run spread. Decode is unaffected: `ids->ne[1] <= 8` takes the mat-vec path; tg32 16.48 vs
16.25, also inside the spread.

## 2b. Separate observation: pp512 is far below the documented baseline in this tree

The same control run reports pp512 158 and tg32 16.5, while AGENTS.md records 349.93 and 17.7-19.1
for the same `-ncmoe 27 -fa on` settings. pp2048 ub2048 matches (565 vs 578, inside the +-3%
drift), so the gap is specific to ubatch 512. `LLAMA_MOE_PREFETCH=0` does not recover it
(146.65 +-26.03), and the value sits on top of the documented `-nopo 1` result (169.32), which
suggests the host MoE layers are computed on the CPU instead of uploaded during a 512-token
prefill. The working tree carries uncommitted changes to `llama-graph.cpp`, `llama-moe-cache.cpp`
and the Vulkan prefetch API, so this is most likely in that in-progress work. It is not caused by
this change (the BK4 toggle is inside the spread, and the s tile path is untouched). Needs its own
investigation before the next baseline refresh.

## 3. Why it regresses: LDS occupancy, not barriers

Derived from the shmem model in `ggml_vk_matmul_int_shmem_support` for the AMD id m tile
(BM=64, BN=64, q4_K: block_a 20 B, block_b 36 B):

| variant | load buffers | row ids | total LDS | workgroups per 32 KB CU |
|---|---:|---:|---:|---:|
| BK_STEP=1 | 3584 | 256 | **3840** | 8 |
| BK_STEP=4 | 14336 | 256 | **14592** | **2** |

The device reports `shared memory: 32768`, so BK_STEP=4 costs three quarters of the residency.
Four k-blocks per iteration remove barriers, but the m tile at n=64 already has enough work in
flight that the lost occupancy dominates.

This is the same mechanism P3 recorded, with a much smaller penalty. P3 sized the shmem arrays
for BK_STEP_MAX=4 on every id tile while still running the m/l tiles at step 1, so it paid the
4x LDS and got none of the barrier savings: 5.11 -> 4.51 TF at batch 2048 (-12%). With real
per-tile SPIR-V blobs the m tile does get the barrier savings, and the net cost is 1.36%. So the
P3 regression was mostly the allocation waste, not the four-k-block loop.

The l tile would be worse still: BM=128, BN=128 gives 29184 B for q4_K and 37376 B for q6_K,
i.e. q6_K would exceed the 32 KB limit and disable the l tile outright.

## 4. Correctness

| suite | result |
|---|---|
| `test-backend-ops -o MUL_MAT_ID -b Vulkan1` (default) | 939/939 passed |
| `test-backend-ops -o MUL_MAT_ID -b Vulkan1` (`GGML_VK_ID_BK4=1`) | 939/939 passed |
| `test-backend-ops -o MUL_MAT -b Vulkan1` (default) | 2275/2275 passed |

The BK_STEP change is numerically neutral by construction: both variants accumulate the k-blocks
into `sums[]` in ascending k order inside the same loop, so BK_STEP only changes how many blocks
are prefetched per barrier pair. No KLD/PPL gate is needed; the operator suite confirms it.

## 5. Outcome

- Default unchanged: s tile BK4, m/l tiles BK1. pp2048 back at the pre-change baseline (565).
- `GGML_VK_ID_BK4=1` keeps the full-BK4 wiring for a device with 64 KB LDS, where the m tile
  would keep 4 workgroups instead of 2. Worth a sweep there (RDNA3/4 report 65536).
- Next lever on this device is not BK_STEP but tile shape: P6 measured BN=32 tiles at 7.40 TF
  at batch 2048 and was blocked only on correctness. A BK_STEP=2 variant (LDS 7680, 4
  workgroups) is a cheap intermediate to try first.

## 6. Repro

```
# baseline (default)
llama-bench -m <model> -dev Vulkan1 -ngl 99 -ncmoe 27 -fa on -t 5 -p 2048 -n 0 -ub 2048 -r 3
# full BK4 on the m/l tiles
set GGML_VK_ID_BK4=1 && llama-bench ... (same arguments)
```
