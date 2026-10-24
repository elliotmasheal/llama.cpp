# qrt - the qwen36 MoE runtime

This directory holds the runtime that replaces the fork's decode path (design section 3.3).
It links neither ggml nor llama: the runtime parses the GGUF itself and drives Vulkan directly.

## Status: M2-0 scaffold

What exists now:

- `qrt_config.*` - the ten switches of design section 7, their design defaults, strict
  validation, and non-fatal notes for legal-but-costly values.
- `qrt_telemetry.*` - a CSV sink with one row per decode pass, holding the columns
  `bench/README.md` section 2 asks for.
- `qrt_main.cpp` - the `qrt` binary: `--help`, `--print-config`, `--selftest`.
- `CMakeLists.txt` - the `qrt_core` library and the `qrt` executable, plus the
  `qrt_selftest` ctest case.

What does not exist yet: GGUF indexing (M2-1), the Vulkan device and allocator (M2-2),
weights and the expert cache (M2-3), kernels (M2-4), prefill (M2-5), decode (M3).
`runtime/proto/` and `runtime/cpu_moe*` predate the scaffold and keep their own targets.

## Build

The target is on by default (`LLAMA_BUILD_QRT=ON`); it needs no Vulkan headers.

```
cmake -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan --target qrt -j 8
```

## Run

```
build-vulkan/bin/qrt.exe --help            # switch table with the design defaults
build-vulkan/bin/qrt.exe --print-config    # effective values, one key=value per line
build-vulkan/bin/qrt.exe --selftest        # switches, error paths, telemetry round trip
```

`--selftest` clears the `QRT_*` variables it tests, so it does not depend on the caller's
environment. It is also registered as the `qrt_selftest` ctest case.

Exit codes: `0` success, `2` bad usage or an unusable telemetry sink (nothing has been
allocated or opened, so the caller can fix the flags and rerun), `1` selftest failure.
No code path aborts on a bad value.

Precedence: CLI flag > `QRT_*` environment variable > design default. A flag takes its
value either as `--flag value` or `--flag=value`; the last occurrence wins.

Notes are not errors. Values that are legal but change the design's cost model (for
example `--prefill-chunk 512`, `--kv-precision f16`, `--draft-vocab 0`) print a `note:`
line on stderr and keep running.

## Telemetry

`--telemetry 1 --telemetry-path <file>` writes a CSV with this header:

```
pass,pos,k,accepted,hit_experts,miss_experts,miss_cpu,miss_h2d,h2d_bytes,draft_ms,gpu_ms,cpu_leg_ms,pass_ms
```

- `pass` is the decode-pass index, `pos` the position of the first verified token.
- `miss_cpu` and `miss_h2d` split `miss_experts` by the arm that served it, so the
  adaptive split (design 4.7) can be checked against the three-leg cycle.
- The file is line buffered and every row is flushed, so a crash keeps the measured rows.
- A pass with telemetry off calls `qrt_telemetry_pass(NULL, ...)`, which returns after one
  branch; the sink is never opened.
- `--telemetry-path` is a runtime extension, not a design switch: it only says where the
  sink goes. `bench/raw/` is the place for measurement runs.

## Files

| file | role |
|---|---|
| `qrt_config.h/.cpp` | switch table, defaults, parsing, validation, notes |
| `qrt_telemetry.h/.cpp` | per-pass CSV sink |
| `qrt_main.cpp` | CLI entry and `--selftest` |
| `CMakeLists.txt` | `qrt_core`, `qrt`, `qrt_selftest` |
