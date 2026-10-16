# Tuning service

A dedicated tuner that measures a (hardware, model) pair once, fits a model of prefill
throughput as a function of the ubatch, and persists the best launch config so it can be
queried cheaply and refined as the machine is used.

## Why

A fixed `-ub 2048` is wrong in general. The right ubatch depends on the GPU (bandwidth,
compute), the model (weight bytes, active params) and the VRAM budget. This service derives
it from measurements instead of hardcoding it.

## The model it fits

For a single prefill pass of `M` tokens (prompt = ubatch = M), prefill throughput is:

```
T(M) = M / (A + B*M)          [tokens/s]
1/T  = B + A*(1/M)            -> linear in 1/M, least-squares fit
```

- `A = weight_bytes / BW_eff`  : time to read all weights once (independent of M)
- `B = flops_per_token / FLOPS`: per-token compute time
- knee `M* = A/B`              : past it the pass is compute-bound
- from the fit: `BW_eff = weight_bytes / A`, `FLOPS_eff = flops_per_token / B`

The recommended ubatch is the smallest grid value `>= M*` that actually ran (i.e. on the
compute plateau); it is bounded by VRAM and by attention cost at large prompts.

## Files

| file | role |
|---|---|
| `tune.ps1` | measure + fit + persist one (hardware, model) entry |
| `best.ps1` | print the stored best args for a model (or all) |
| `service.ps1` | background loop: tune every new model, refresh periodically |
| `fit-ubatch.ps1` | offline: fit A/B from existing `results/P3fit_M*` logs |
| `store/<gpu>__<model>.json` | persisted profiles (identity, fit, best, history) |

## Use

```powershell
.\tune.ps1 -Model F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf
.\best.ps1 -Model F:\LLM\llm\Ornith-1.5-35B-Q4_K_M.gguf
.\service.ps1            # leave running; tunes new models as they appear
```

`tune.ps1` measures, in order:
1. residency (`-ncmoe`) sweep at a small ubatch -> best ncmoe by whole-turn wall time
2. PP(ubatch) single-pass sweep at that ncmoe -> fit A, B, knee, BW_eff -> best ubatch
3. residency refine at the chosen ubatch (ubatch and ncmoe interact)
4. final measure of the chosen (ubatch, ncmoe)

and writes everything to the store.

## Interaction to know about

ubatch and `-ncmoe` (residency) are not independent: the best residency measured at ub 512 can differ
from the best at ub 2048. That is why the service re-sweeps residency after picking the ubatch (step 3).
Some ubatch values also land on pathological code paths (a non-monotonic point was measured at M=1024 with
one residency setting), so the fit should be eye-checked against the raw `pp_points` in the store.
