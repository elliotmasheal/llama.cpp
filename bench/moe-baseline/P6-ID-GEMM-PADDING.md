# P6: MUL_MAT_ID MoE GEMM at prefill batch >= 2048 - measured state and the blocker (2026-10-21)

Base HEAD `c707d5633`. No code change kept: the one configuration that reaches the per-op target
(BN=32 tiles, 7.40 TF at batch 2048) still produces wrong results on part of the correctness
suite, so it is not shippable yet. This doc records the measurements and the exact blocker.

## 1. Per-op baseline (test-backend-ops perf, 256 experts top-8, unaligned)

| op | batch 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| q4_K gate/up (m=512 k=2048) | 4.02 | 4.73 | 5.11 | 5.59 | 5.81 | 6.04 |
| q4_K down (m=2048 k=512) | 3.72 | 4.59 | 5.11 | 5.52 | - | - |
| q6_K gate/up | 2.60 | 3.46 | 4.83 | 5.68 | 6.17 | 6.61 |
| q6_K down | 2.59 | 3.51 | 5.03 | 5.73 | - | - |

Dense references for the same byte counts and shapes (no routing, contiguous rows):

| dense op | TFLOPS |
|---|---:|
| q4_K m=512 n=2048 k=2048 (one expert GEMM) | 7.91 |
| q4_K m=4096 n=2048 k=2048 (8 experts worth of rows) | 8.93 |
| q4_K m=16384 n=2048 k=512 (8 down experts) | 8.59 |
| q6_K m=512 n=2048 k=2048 | 6.01 |
| q6_K m=16384 n=2048 k=512 | 6.54 |

So the id path is at 65% (gate/up) and 60% (down) of the dense path at batch 2048, and still at
76% at 16384 where tile padding is minimal. The remainder is structural to the id kernel plus the
padding of each expert's last token tile.

## 2. Why the id path is slower: per-expert tile padding

A MUL_MAT_ID tile covers one expert and BN tokens of that expert; the tile count per expert is
ceil(tokens_e / BN). At batch 2048 the average is 64 tokens per expert, so with BN=64 roughly half
the experts need a second, nearly empty tile (average about 7 valid rows out of 64). That second
tile re-reads all BM weight rows of its expert and runs a mostly idle compute loop, which is the
main source of the gap.

## 3. Tile sweep (all configs measured, m-tile replaced, batch 2048 / 4096)

| m-tile (BLOCK, BM, BN, WM, WN, WMITER, TM, TN, WARP) | 2048 | 4096 | correctness |
|---|---:|---:|---|
| current {128, 64, 64, 32, 32, 1, 2, 2, 32} | 5.11 | 5.59 | 939/939 |
| {128, 128, 64, 32, 64, 1, 2, 2, 32} | 2.63 | 2.88 | 939/939 |
| {256, 128, 64, 32, 32, 1, 2, 2, 32} | 2.61 | 2.87 | 939/939 |
| {256, 64, 128, 32, 32, 1, 2, 2, 32} | 4.41 | 4.85 | 939/939 |
| {512, 128, 128, 32, 32, 1, 2, 2, 32} | 1.80 | 1.94 | 939/939 |
| **{64, 64, 32, 32, 32, 1, 2, 1, 32}** | **7.40** | **8.99** | **64 FAIL** |
| {64, 64, 128, ...} (warps mismatch, invalid) | 6.35 | 7.64 | fail (expected) |

The BN=32 tile removes the padding waste and reaches 7.40 TF at batch 2048, 8.99 at 4096, 10.53 at
16384 (q6_K 6.14 / 7.71 / 9.78), i.e. the per-op target of 6.3 is reachable, but it fails 64
correctness cases and is therefore not kept.

## 4. The blocker in detail

With the BN=32 m tile the id GEMM fails exactly these eval shapes (same for q2_0, q4_0, q4_1, q8_0,
q4_K, mxfp4): `n_mats=4 n_used=4 m=512 k=256` with `n = 33, 47, 48, 49, 129`. Those are the
token-tile boundary cases: n_used == n_mats so every token routes to every expert, each expert gets
exactly n rows, and the last tile holds 1, 15, 16 or 17 valid rows. n = 32 and n <= 31 pass, and
all shapes with k >= 512 pass. k=256 is also the shape where `ggml_vk_should_use_mmvq` returns
false, so those cases run the f16/f32 B path instead of the q8_1 MMVQ path while the model
(batch >= 2048, k=2048/512) uses both paths.

Entry points for the fix: `load_row_ids()` in `mul_mm_id_funcs.glsl` (writes only `_ne1` entries of
`row_ids[BN]`), the `_ne1` column guards in `mul_mmq.comp` (lines around 213 and 248), and
`ggml_vk_mul_mat_id_max_tiles()` (tile upper bound). Until that is understood, BN must stay 64.

## 5. End-to-end on the unchanged tree (recommended config: -b 4096 -ub 4096 cache 4608)

| case | target | measured |
|---|---:|---:|
| 10k prompt prefill, ctx 32768 | >= 460 | 448.6 |
| 20k prompt prefill, ctx 32768 | >= 365 | 365.5 |
| 10k prefill, 256K ctx (ub 2048, no cache) | >= 372 | 370.8 |
| 20k prefill, 256K ctx | >= 320 | 319.6 |
| pp512 (cache 6144) | >= 395 | 398.51 +-3.42 |
| pp2048 ub 2048 | >= 575 | 576.84 +-1.43 |
| tg32 (cache 6144) | >= 26.4 | 26.18 +-0.43 |
| decode at 10k / 20k long ctx | >= 20.6 | 20.3 / 19.6 |

The remaining gap to 460 is the MoE GEMM: at ub 4096 the MoE gate/up GEMM runs at 5.59 TFLOPS.
Fixing the BN=32 boundary failure and keeping 7.4-9.0 TFLOPS at batch 2048-4096 should move 10k
prefill past 500 t/s, so the blocker is worth a dedicated follow-up.
