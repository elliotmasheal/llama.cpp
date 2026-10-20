# P4: dual id SPIR-V variants - BK_STEP=4 for the AMD s tile only (2026-10-20)

Base HEAD `069eeecb5`. Changed: `ggml-vulkan.cpp`, `mul_mmq.comp`, `vulkan-shaders-gen.cpp`.

## Change

`mul_mmq.comp` keeps BK_STEP as a compile-time define (id default 1, dense 4). The generator now
also emits an id-only variant with `BK_STEP=4` (`..._bk4`). On AMD non-GCN devices the id s tile
pipeline is created from the bk4 blob while the m/l tiles keep the default blob, so the 4x load
buffers are only paid where they buy something. `create_mm_pipelines` gained a tile offset (the
pipeline vector no longer assumes it is filled from index 0) and the id s-tile shmem check uses
the 4x size. Other vendors and the dense path are unchanged; the m/l tiles keep the small LDS.

## Per-op (test-backend-ops perf, 256 experts top-8)

| op | before (069eeecb5) | after |
|---|---:|---:|
| q4_K gate/up m=512 n=512 k=2048 | 3.15 TF | **3.97 TF** (+26%) |
| q4_K down m=2048 n=512 k=512 | 3.14 | **3.69** (+17%) |
| q6_K gate/up | 2.51 | 2.60 |
| q6_K down | 2.58 | 2.57 |
| q4_K batch 2048 gate/up, down | 5.11, 5.11 | 5.11, 5.13 |
| q6_K batch 2048 gate/up, down | 4.81, 5.03 | 4.82, 5.03 |
| A-stream probe q4_K / q6_K (n_used=1) | 1.99 / 2.59 ms | 1.49 / 2.41 ms |

## End to end

| case | before | after |
|---|---:|---:|
| cache6144 pp512 | 387.17 +-5.75 | **401.21 +-3.97** |
| cache6144 tg32 | 26.15 +-0.38 | 26.57 +-0.24 |
| cache6144 pp2048 ub2048 | 578.31 +-3.40 | 577.51 +-1.09 |
| long-ctx 10k ub2048, ctx 32768 | 410.8 / 410.6 | 411.3 |
| long-ctx 20k ub2048, ctx 32768 | 346.7 / 347.0 | 346.6 |
| long-ctx 10k / 20k ub1024 | 347.4 / 299.3 | 352.2 / 302.5 |
| 256K ctx 10k / 20k prefill | 372.2 / 320.5 | 371.6 / 320.5 |
| decode 10k / 20k / 256K | 20.5 / 20.2 / 14.5 | 20.8 / 20.6 / 14.3 |

ub1024 still does not beat ub2048 for long prompts (upload per ubatch dominates), but it gained
~1.4% from the faster s tile.

## Correctness

- test-backend-ops MUL_MAT_ID Vulkan1: 939/939 OK (exercises every bk4 id type)
- MUL_MAT_ID CPU: 945/945 OK; MUL_MAT Vulkan q4_K/q6_K: 151/151 OK
- FLASH_ATTN_EXT Vulkan1: 5332/5332 OK
- text: byte-identical to the pre-change build for the same prompt/seed (only the build-id line
  differs), PPL 1.0077 both builds
