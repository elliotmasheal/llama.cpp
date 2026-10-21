# P6 correction: the BN=32 m-tile "win" was a measurement artifact (2026-10-21)

Follow-up to `P6-ID-GEMM-PADDING.md`. The boundary failures recorded there were not a kernel bug,
and the +45% was not real. Both came from an inconsistent experiment. Conclusion: the 64-token m
tile stays and no code change is kept.

## 1. Root cause of the 64 "boundary" failures

`ggml_vk_load_shaders` builds three id tile vectors:

- `tc_mmqid`   (id GEMM with f16/f32 B, shader `mul_mm.comp`) uses `m_warptile_mmqid` +
  `m_mmqid_wg_denoms` for the m slot.
- `tc_mmqid_int` / `_int_k` (id GEMM with q8_1 B, shader `mul_mmq.comp`) use
  `m_warptile_mmqid_int` + **`m_mmq_wg_denoms`** for the m slot.

The experiment set the int warptiles to a BN=32 tile but only changed `m_mmqid_wg_denoms`. So the
int pipelines ran a shader with BN=32 while the launch (row_map tile size, grid y) was built with
BN=64, and the f16 pipelines ran a shader with BN=64 while the launch used BN=32. Both are
inconsistent, so the results were wrong. The failing shapes were simply the ones where the
mismatch produced a detectable error (f16 path: `n_mats=4, k=256, n=33/47/48/49/129`).

The kernel's own boundary handling is fine: with the same warptile and both launch vectors moved
together, all 939 MUL_MAT_ID Vulkan cases pass.

## 2. A correct BN=32 tile is slower, not faster

Change tried (kept consistent, scoped so the f16/dequant path stays on BN=64, which is also
required for the iq-shared-memory init in `mul_mm.comp`):

```
m_warptile_mmqid_int = m_warptile_mmqid_int_k = {128, 64, 32, 32, 32, 32, 1, 2, 1, 1, 32};  // BLOCK,BM,BN,...
m_mmqid_wg_denoms = {64, 32, 1};
and the four int id tile vectors switched from m_mmq_wg_denoms to m_mmqid_wg_denoms
```

Correctness: 939/939. Performance, however, regresses:

| op | current BN=64 | consistent BN=32 |
|---|---:|---:|
| q4_K gate/up batch 2048 | 5.11 | 4.89 |
| q4_K down batch 2048 | 5.11 | 4.91 |
| q4_K gate/up batch 4096 | 5.59 | 5.22 |
| q4_K down batch 4096 | 5.52 | 5.13 |
| q6_K gate/up batch 2048 | 4.83 | 4.37 |
| q6_K down batch 2048 | 5.03 | 4.50 |

Reason: a token tile of 32 columns halves the number of tokens each 64-row weight tile is used
for, so every weight row is streamed twice as often. The id op at these batches already streams
151 MB of weights per call, and that extra traffic costs more than the saved padding compute.

The earlier numbers (7.40 TF at batch 2048, 8.99 at 4096, 10.53 at 16384) were measured with the
mismatched configuration, where each workgroup computed only half of its token columns, so they
are invalid.

## 3. Where the remaining id GEMM gap actually is

With the current tile the id GEMM asymptote is 6.04 TFLOPS (q4_K) and 6.61 (q6_K) at
16384 tokens, versus 7.91 and 6.01 on the dense path for the per-expert shape and 8.93 for the
8-experts-worth dense shape. Padding is only a small part of it: at 16384 tokens each expert has
512 tokens, so tiles are full, yet the id path is still 24% behind dense q4_K. The remaining
difference is per-tile overhead that the dense path does not pay: the row_map descriptor load, the
expert base computation, the gathered activation rows (each token row is read once per m tile),
and no reuse of weights across experts.

## 4. State kept

No code change. End-to-end numbers are the ones in `P6-ID-GEMM-PADDING.md` section 5
(10k 448.6, 20k 365.5, 256K 370.8 / 319.6, pp512 398.51, pp2048 576.84, tg32 26.18). Correctness on
the unchanged tree: MUL_MAT_ID Vulkan 939/939, CPU 945/945.
