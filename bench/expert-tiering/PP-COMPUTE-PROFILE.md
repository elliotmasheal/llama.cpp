# PP compute profile (pp2048, prefetch on, 2026-10-22)

Session state: pp512 335, pp2048 523 (this machine runs ~25% below the AGENTS numbers; compare
within a session only). Note: back-to-back pp2048 runs heat the card and later runs of the same
command can drop to ~310 t/s; leave ~40 s idle between runs or use r=3 with low variance.

## Where the 3.9 s go

`GGML_VK_PERF_LOGGER=1` + `GGML_SCHED_TRACE=1`, pp2048 ub2048, `-ncmoe 41 --moe-cache-mib 6144`:

- eval wall 3.9 s, GPU busy (kernel time summed by the perf logger) 3.88 s -> **the GPU is not
  idle and the prefetch is not on the critical path**: the host wait for the prefetch is 2.0 ms per
  layer (the first layer exposes 37 ms), and a run with the upload copies skipped (`NOOP` probe,
  timing only) gives the same 516 t/s. So PP is compute bound, not transfer bound.
- Per layer wall is a flat ~95 ms, so vectorized/large copies are not the issue.

Per eval kernel time (sum of the perf logger tables, two evals; kb = 11 layers of 2048? no: 30 GDN,
10 attention):

| op | ms/eval | share | notes |
|---|---:|---:|---|
| CONCAT (30x, dim-0) | 751 | 19% | `conv_input = concat(conv_states [3,8192], qkv_mixed^T [2048,8192])` = 67 MB per GDN layer, ~25 ms each |
| MUL_MAT_ID (MoE, 32 prefetch layers) | 899 | 23% | q4_K gate/up 3.5-3.7 TFLOPS, q6_K down 3.5 |
| FLASH_ATTN_EXT (10 attention layers) | 359 | 9% | 36 ms per layer, long-ctx memory bound |
| MUL_MAT (dense, expert/shared/attn projections) | 955 | 25% | q4_K 6.3-6.6 TFLOPS, q6_K 3.6 |
| GATED_DELTA_NET | 153 | 4% | |
| RMS_NORM_MUL / MUL / ADD / GLU / SSM_CONV | ~270 | 7% | |
| SUM | 3.88 s | | |

## The concat is the first target: it is a transposed read, not a copy

`qkv_mixed` is `[channels=8192, tokens=2048]`; the graph transposes it and concatenates 3 state rows
in front, producing `[2051, 8192]` for the Mamba/DeltaNet short conv (time is dim 0). The generic
`concat.comp` walks the destination in order, so the 2048 values of one channel row are read at a
32 KiB stride: every 4-byte element pulls a 32-byte sector (8x amplification) and the kernel reaches
only 2.7 GB/s (67 MB in 25 ms). The transpose node itself is free (a view); the cost all lands in
`CONCAT`.

Fix options, in order of expected value:

1. **Tiled transpose + concat shader** (new compact kernel): 32x32 shared-memory tiles, read
   `qkv_mixed` coalesced and write `conv_input` coalesced; both become one streaming pass at
   memory speed (~1-2 ms per layer). Expected: 751 -> ~50-100 ms/eval, i.e. eval 3.9 -> 3.2-3.3 s,
   pp2048 ~620-640 t/s.
2. **Remove the concat from the graph** and feed the conv from `qkv_mixed` directly: the conv is
   linear, so out[t] = sum_j w[j] * qkv[t-3+j] for t >= 3 and only out[0..2] mix the state; a
   strided `conv_input` view of `qkv_mixed` (time contiguous, channel stride 8192) is exactly what
   `ssm_conv.comp` already reads (`i2 + i0 + i1*nb01/4`), so the main conv can run on a view and the
   3 boundary rows can be computed with elementwise ops. Bigger graph surgery, no new kernel.
3. Leave it alone and go for the MoE GEMM (899 ms, 3.5 TFLOPS vs 6.6 dense) - a worse ratio.

## Next steps (PP compute)

1. Implement option 1 (tiled transpose+concat) - contained in one shader plus one dispatch, and
   numerically exact (same values, only the access pattern changes).
2. Re-profile, then attack the MoE GEMM (MUL_MAT_ID q4_K/q6_K on the bank view) and the dense q6_K
   matmuls; those are the remaining ~1.85 s.
3. 780 t/s needs eval <= 2.63 s, i.e. -1.28 s from the current 3.9 s: concat (~0.7 s) plus MoE GEMM
   and dense improvements (~0.6 s) are the required combination.
