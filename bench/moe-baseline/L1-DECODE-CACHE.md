# L1: decode expert cache - coverage, prefill, acceptance

Date: 2026-10-18. Branch `moe-gpu-cpu-hybrid`. Machine: RX 6700S 8 GB, `-dev Vulkan1 -ngl 99 -fa on -t 5`.
All numbers are llama-bench `-r 3`, interleaved A/B, static `-ncmoe 27` vs cache
`-ncmoe 41 --moe-cache-mib 6144` unless noted. `--moe-hybrid` stays gated off.

## Coverage: the "20/41 layers" reading was wrong

The cache printed one line per *layout group*: `20 layers, q6_K: 1690 slots` and
`20 layers, q4_K: 1690 slots`. Down projection of the first/second layer set is Q6_K vs Q4_K, so
the 40 MoE layers split into two groups of 20; together they cover every layer. With `-ncmoe 41`
the new log says `40 of 40 MoE layers have a bank`. Coverage was never the problem: at 6144 MiB the
bank holds 33% of the 18.6 GiB of host experts, so on a uniform router the hit rate is about the
coverage. Nothing doubles the hit rate at a fixed bank size.

## What actually moves the numbers

1. Prefill time is dominated by the per-layer host->device expert copies and the scheduler wait
   before each copy (trace at pp2048 ub512: 81 copies / 2.3-3.2 s `cpy_sync` for `-ncmoe 27`,
   97-108 copies / 8.1-8.8 s for the cache). A partially resident layer still pays 3 full host
   copies, so only a *fully resident* layer removes them.
2. Decode benefits while the slots are spread over all layers (temporal locality): the warmed LRU
   reached 48% hits at 33% coverage.
3. New layout: the first `n_full` layers of each group keep every expert in the bank (no host copy
   ever), the remaining slots are the LRU pool of the other layers. Measured pp512 vs whole layers
   per group (llama-bench r=1): 0/2/3/4/5/6 -> 291/318/328/344/356/369 t/s; 5 is the chosen point
   (1690 slots per group = 1280 whole-layer + 410 LRU slots).

## Acceptance measurements (r=3, interleaved A/B, 2 rounds agree within 1%)

| test | static `-ncmoe 27` | cache `-ncmoe 41 --moe-cache-mib 6144` | delta |
|---|---:|---:|---:|
| tg32, `-ub 512` | 18.02 | 24.09 | +33.7% |
| tg32, `-ub 2048` | 18.97 | 24.79 | +30.7% |
| pp512, `-ub 512` | 344.5 | 358.0 | +3.9% |
| pp2048, `-ub 2048` | 576.0 | 578.3 | +0.4% |
| pp2048, `-ub 512` | 163.9 | 59.0 | -64% |

At `--moe-cache-mib 4096` the whole-layer reservation is capped by the smaller bank: tg32 22.6 and
pp512 328 (safety rule: leave at least 1/5 of the slots to the LRU pool). 4096 is therefore below the
pp512 target; 6144 is the recommended size.

TG >= 23, PP512 >= 340 and pp2048 ub2048 are met. pp2048 with `-ub 512` regresses: the whole-layer
residency makes the scheduler's copy-buffer reuse pattern alternate, so the wait before each host
weight copy grows from ~29 ms to ~83 ms (trace: `cpy_sync` 2.3-3.2 s vs 8.1-8.8 s per ubatch). With
`-ub 2048` the prompt is a single ubatch and the regression disappears. Canonical config:
`-ncmoe 41 --moe-cache-mib 6144 -ub 2048 -t 5` (decode is unaffected by `-ub`).

## Correctness

- cache vs `-ncmoe 41` (same placement), with the decode MoE forced onto the GPU in both runs
  (`GGML_OP_OFFLOAD_MIN_BATCH=1`), 514 token prompt, `-n 32 --temp 0`: **byte-identical text**.
  This is the meaningful equality: same weights, same GPU kernels, only the source of the device
  copy differs.
- cache vs static `-ncmoe 27`: greedy text differs, because static decodes 27 layers on the CPU
  and the cache decodes every layer on the GPU. This is the known quantized-kernel numerics effect
  (all-CPU PPL 6.1161 vs GPU cache PPL 6.1973 measured earlier), not a cache error.

## Files

- `src/llama-moe-cache.cpp`: whole-layer residency + LRU pool (`slot_base`), one-time bank
  pre-population at init, per-group coverage log.
- `tools/llama-bench/llama-bench.cpp`: `--moe-cache-mib` argument.
