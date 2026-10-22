"""Reads the LLAMA_MOE_STATS file (decayed per-expert selection weights).

File layout (native little endian): u32 magic 'MOES' (0x53454f4d), u32 version, u32 n_layer,
u32 n_expert, then n_layer*n_expert f32 weights.

Usage:
  python read_stats.py <stats.bin>                 summary + per-layer top experts
  python read_stats.py <stats.bin> --permute <json>  write the hot-first permutation per layer
"""
import json
import struct
import sys

MAGIC = 0x53454F4D


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, n_layer, n_expert = struct.unpack_from("<IIII", data, 0)
    if magic != MAGIC or version != 1:
        raise SystemExit(f"unexpected header: magic=0x{magic:08x} version={version}")
    n = n_layer * n_expert
    weights = struct.unpack_from(f"<{n}f", data, 16)
    return n_layer, n_expert, list(weights)


def gini(vals):
    s = sorted(vals)
    tot = sum(s)
    if tot <= 0:
        return 0.0
    n = len(s)
    return sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(s)) / (n * tot)


def main():
    path = sys.argv[1]
    out_json = sys.argv[sys.argv.index("--permute") + 1] if "--permute" in sys.argv else None
    n_layer, n_expert, w = load(path)
    print(f"file: {path}")
    print(f"layers: {n_layer}, experts: {n_expert}, total weight: {sum(w):.1f}")

    perm = {}
    for il in range(n_layer):
        row = w[il * n_expert:(il + 1) * n_expert]
        tot = sum(row)
        if tot <= 1e-6:
            continue
        ranked = sorted(range(n_expert), key=lambda e: (-row[e], e))
        top8 = sum(row[e] for e in ranked[:8]) / tot
        top32 = sum(row[e] for e in ranked[:32]) / tot
        perm[il] = ranked
        if il % 5 == 0 or il in (0, n_layer - 1):
            print(f"layer {il:2d}: gini={gini(row):.3f} top8={top8:.3f} top32={top32:.3f} "
                  f"top={ranked[:6]}")

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"n_layer": n_layer, "n_expert": n_expert, "perm": perm}, f)
        print(f"permutation written to {out_json} ({len(perm)} layers)")


if __name__ == "__main__":
    main()
