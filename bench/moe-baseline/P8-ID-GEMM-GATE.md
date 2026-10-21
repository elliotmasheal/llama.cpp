# P8 gate: structural id-GEMM rewrite (analysis only, not implemented) (2026-10-21)

Base HEAD `2f0d106d7`. Task-32 asked for an analysis gate before any shader rewrite: if the best
expected end-to-end gain is below 8% (10k prompt < 485 t/s), stop. The measured share of the MoE
id GEMM is 18.4% of prefill time, so the gate fails and no code was changed.

## 1. Measured share of the id GEMM in prefill

Direct probe: run the 10k prefill with the id GEMM's k work temporarily halved (wrong results, a
timing probe only) and compare with the unchanged build, same config
(`-ncmoe 41 --moe-cache-mib 4608 -b 4096 -ub 4096 -c 32768 -t 5`):

| build | 10k prefill |
|---|---:|
| normal | 448.8 / 448.9 |
| id GEMM k halved | 489.9 |

Halving the id GEMM work gains 9.2%, so the id GEMM is 2 x 9.2 = **18.4% of prefill wall time**
(the remaining 81.6% is dense GEMM, flash attention, upload and the per-layer sync chain).

In-situ rates match the isolated microbenchmarks: during prefill the MoE GEMM runs the same shapes
(`m=512 k=2048` gate/up, `m=2048 k=512` down, 4096 tokens) at 5.5-5.6 TFLOPS, i.e. what
`test-backend-ops` reports, so the tile behaviour in situ is not different.

## 2. Gate arithmetic

| scenario | id GEMM speedup | prefill gain | 10k t/s |
|---|---:|---:|---:|
| realistic structural rewrite (5.55 -> 6.8 TF) | 1.22x | +3.3% | 464 |
| full dense parity (5.55 -> 8.05 TF) | 1.45x | +5.7% | **474** |
| needed for the 485 gate | 1.78x | +8.1% | 485 |
| id GEMM made infinitely fast | inf | +18.4% | 531 |

Dense parity is 474 t/s, below the 485 threshold, so even a rewrite that removes every bit of the
id GEMM's structural overhead (which cannot happen: the id GEMM is 256 small per-expert GEMMs and
the dense reference is one large one) fails the gate. Reaching 485 requires 9.9 TFLOPS on the id
path, above the dense path's own 7.9-8.6.

## 3. Expected gain of each candidate mapping

(a) Expert dimension folded into the grid, weight tile carried across all token tiles of the
expert. At ub 4096 each expert has 128 tokens, so with BN=64 there are two token tiles per expert
and the 151 MB of expert weights is streamed twice. Serving both token tiles from one weight tile
removes 50% of that traffic: 75 MB at 224 GB/s = 0.34 ms out of the 12.4 ms gate/up op = 2.7% per
op, i.e. **+0.5% end to end**. Splitting k would keep occupancy but adds a reduction pass.

(b) Cross-expert activation reuse / persistent activation path. The activation tensor is 4096 x
2048 in q8_1 = 8.9 MB, which fits the 32 MB infinity cache, and every token's row is re-read 8
experts x 8 m tiles = 64 times from that cache. There is no DRAM traffic to save, so the expected
gain is ~0; it could only help L2 bandwidth, which is not the limiter at 5.5 TFLOPS.

(c) Merged-experts GEMM (top-8 as one wider N). The eight experts have disjoint weight matrices, so
merging the N dimension means a block-diagonal weight tile and 8x wasted MACs. Infeasible without
changing the arithmetic.

Best realistic option (a): +0.5% end to end. The gate's 8% is out of reach.

## 4. Conclusion

The id GEMM is not the remaining prefill bottleneck. At 18.4% of prefill wall time, even making it
free only reaches 531 t/s; the structural rewrite targets a per-tile overhead that is worth well
under 1% end to end. The next lever is the other 81.6%: the dense GEMM (attention/SSM/shared
projections), flash attention, and the upload + per-layer synchronization chain.

No code change, no shader rewrite. The documented fallback stays: 10k prompt 448.6 t/s, 20k 365.5,
256K 370.8 / 319.6, pp512 398.5, pp2048 576.8, tg32 26.18, long-context decode 20.3 / 19.6.
