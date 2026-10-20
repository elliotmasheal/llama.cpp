# P3: per-tile BK_STEP for the id GEMM, and the long-context ubatch sweep (2026-10-20)

Workspace `E:\fastLlama\.worktrees\moe-hybrid`, base HEAD `a5bb899e8` (= master after the P2 merge).
No code change was kept: the per-tile BK_STEP experiment is measured but unsafe (see below), and
the long-context question is answered by the sweep.

## Part 1: per-tile BK_STEP (spec constant 11 + fixed-size shmem arrays) - NOT kept

Implementation tried: `mul_mmq.comp` takes BK_STEP as specialization constant 11 (default 4) and
sizes `buf_a`/`buf_b` for BK_STEP_MAX=4; the id int tile vectors get the step appended per tile
(s tile 4 on AMD else 1, m/l 1). Shmem check updated to the array size (4).

Per-op (test-backend-ops perf, 256 experts top-8):

| op | before (a5bb899e8) | per-tile BK_STEP |
|---|---:|---:|
| q4_K gate/up m=512 n=512 k=2048 | 3.15 TF | **3.97 TF** (target 3.9 met) |
| q4_K down m=2048 n=512 k=512 | 3.14 | 3.69 |
| q6_K gate/up / down | 2.51 / 2.58 | 2.58 / 2.56 |
| q4_K batch 2048 gate/up / down | 5.11 / 5.11 | **4.51 / 4.59** |
| q6_K batch 2048 gate/up / down | 4.81 / 5.03 | **3.67 / 3.79** |

Why it regresses batch 2048: the fixed-size arrays are allocated for every id tile.
`GGML_VK_PIPELINE_STATS` reports `ldsUsageSizeInBytes: 14848` for the m tile (BK_STEP=1) versus
about 3.7 KB before. On this 32 KB-LDS device that drops residency from ~8 to 2 workgroups per CU,
so the m tile (used for tokens/expert in 33..64, i.e. ub 2048) loses 12-25%.

Safe path (not done, needs more plumbing): compile the id int shaders twice, one variant with
BK_STEP_MAX=1 (m/l) and one with BK_STEP_MAX=4 (s), and let the pipeline creation pick the SPIR-V
per tile. That needs a tile-offset in `create_mm_pipelines` (the pipeline vector is filled from
index 0 on the first call) and a second set of generated id SPIR-V blobs.

## Part 2: long-context ubatch sweep (current build, cache5120, ctx 32768)

llama-cli, `-ncmoe 41 --moe-cache-mib 5120 -c 32768 -n 48 -st`, prompt files
`bench/longctx/prompt10k.txt` / `prompt20k.txt`, two interleaved runs per config:

| config | 10k prefill | 20k prefill | decode@10k |
|---|---:|---:|---:|
| ub 512 | 261.8 / 260.6 | 230.7 | 20.6 |
| ub 1024 | 347.4 / 346.8 | 299.3 | 20.4 |
| **ub 2048** | **411.2 / 410.8** | **347.0** | 20.5 |

ub 2048 stays the best: the fast s tile applies at ub 512/1024 (tokens/expert 16/32) but the
smaller ubatch multiplies the per-ubatch expert upload, which dominates (L5/L9 finding). ub 1024
is 15.5% slower at 10k and 13.7% slower at 20k. Target 450 t/s is therefore not reachable through
the ubatch choice; 10k stays at 410-411 and 20k at 347.

## Part 3: regression checks on the unchanged tree

| case | value |
|---|---:|
| cache6144 pp512 | 387.17 +-5.75 |
| cache6144 tg32 | 26.15 +-0.38 |
| 256K ctx, ncmoe 41, 10k prefill | 372.2 (was 370.7) |
| 256K ctx, ncmoe 41, 20k prefill | 320.5 (was 320.7) |
| decode at 10k/20k/256K | 20.5 / 20.2 / 14.5 |

Correctness on the final tree: MUL_MAT_ID Vulkan 939/939, CPU 945/945; MUL_MAT Vulkan q4_K/q6_K
151/151 and FLASH_ATTN_EXT 5332/5332 are unchanged from the accepted a5bb899e8 build.

## Conclusion

The 3.97 TFLOPS s-tile path from earlier is real, but making BK_STEP per-tile with one SPIR-V
costs the m tile more than it gains. The next step is the two-variant SPIR-V route above; it is
the only measured path to keep both (q4_K 3.97 at batch 512 and 5.11 at batch 2048).
