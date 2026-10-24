# qwen36 expert repack format (M5-1)

Status: specification. The writer and verifier are M5-2 (`tools/expert-repack/`); the runtime
reader lands with `weights` (M2-3) and is validated by M5-4.

## Inputs and outputs

- Input: the original GGUF (`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, 22853663008 B), opened read-only
  through `runtime/gguf_index.*`. Tensor table, absolute offsets and byte sizes come from there.
- Output, next to the source:
  - `<model>.qwen36rt.bin` - the repacked expert payload.
  - `<model>.qwen36rt.idx` - the sidecar index (text).
- **No quant value changes.** Every byte of every expert is copied from the GGUF slice as is;
  the writer never dequantizes, requantizes or reorders values inside a slice.

## Source slices

For layer `L` and expert `e`:

| part | tensor | slice |
|---|---|---|
| gate | `blk.<L>.ffn_gate_exps.weight` | `e` of `ne[2]` (ne0*ne1 values, contiguous) |
| up | `blk.<L>.ffn_up_exps.weight` | same |
| down | `blk.<L>.ffn_down_exps.weight` | same |

The expert dimension is the outermost one, so each expert slice is contiguous in the source.
Slice bytes = `ne0 * ne1 / block_size(type) * type_size(type)`. The type mix differs by layer
(the target model has Q4_K gate/up, Q5_K down and some Q6_K), so sizes are **not** constant:
the index carries explicit per-part sizes. The M0 account average of 1917127.8 B per expert is
an average, never a stride.

## `.bin` layout

64-byte header, then the payload:

| field | type | value |
|---|---|---|
| magic | 8 x u8 | `Q36RT01\0` |
| version | u32 | 1 |
| align | u32 | 65536 |
| layers | u32 | 41 |
| experts | u32 | 256 |
| payload_offset | u64 | 64 |
| payload_bytes | u64 | bytes after the header |
| n_experts | u64 | 10496 |
| reserved | 8 x u8 | zero |

Payload: the expert blocks in ascending `(layer, expert)` order. An expert block is
`gate || up || down` back to back, and **each expert block starts at a multiple of 65536
bytes** (zero padding before it). Slices inside a block are not padded, so an H2D copy of one
expert is a single contiguous range.

Alignment rationale: design 4.4 allows 2 MiB or 64 KiB. A 2 MiB stride was sized for a uniform
1.84 MiB expert; with the real type mix it would waste up to ~10 GiB. 64 KiB keeps every
transfer page-aligned, wastes at most ~0.3 GiB in total, and matches the H2D block granularity.

## `.idx` sidecar (text)

```
# qwen36rt index v1
# gguf=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
# gguf_size=22853663008
# gguf_sha256=<hex>
# arch=qwen35moe
# layers=41 experts=256 align=65536
# columns=layer,expert,offset,bytes,gate,up,down,sha256
0,0,0,1920128,589824,589824,740480,<hex>
...
```

- `offset` is relative to `payload_offset`, `bytes = gate + up + down` plus the padding that
  precedes the next block is not counted.
- `sha256` is over `gate || up || down` of that expert, computed from the source GGUF and
  re-checked after the write.
- Loaders must ignore unknown `#` keys and reject a record whose `layer`/`expert` is out of
  range or whose `offset + bytes` exceeds `payload_bytes`.

## Write protocol

1. Write `.bin.partial`, update `.idx.partial` after every expert (append-only), fsync in
   batches. The index is the journal: the last complete record tells the writer where to
   resume after an interruption.
2. On success: write the whole-payload SHA-256 into the index, fsync, then rename
   `.bin.partial` -> `.bin`, `.idx.partial` -> `.idx`. The presence of `.idx` means complete.
3. Any hash mismatch or short read: stop, keep the partial files, exit non-zero. No abort, no
   silent truncation.

## Runtime contract

- `weights` (M2-3) reads the index into `host_offset[layer][expert]` and a size per part,
  maps or opens the `.bin`, and hands the byte range to both the VRAM cache (H2D) and the CPU
  expert core. The CPU core reads the same `gate || up || down` bytes, so one hash covers both.
- If the sidecar is missing or rejected, the runtime must start from the original GGUF and say
  so (M5-4 checks this), never abort.

## Validation (M5-2 / M5-4)

- Per expert: SHA-256 before and after must match (the M5-2 gate).
- Whole payload: one SHA-256 recorded in the index and re-checked by the verifier.
- Two-path greedy: 128 tokens via the repacked file and via the original GGUF must be
  identical (M5-4), and deleting the sidecar must still start the runtime.
- Disk use and repack time are reported in `bench/RESULTS.md`.
