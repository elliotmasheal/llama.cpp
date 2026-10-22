"""Repack a MoE GGUF so the hot experts of every layer come first, using LLAMA_MOE_STATS weights.

Pure permutation: the quantization type, tensor shapes and all metadata stay untouched, so the output
has the same size as the input and the same numerical values. Only the weighted-sum accumulation order
changes (FP32 level), which the quality gate must confirm.

The header, KV and tensor-info section are copied verbatim; only the payload of the routed-expert
tensors and of the router weight is permuted. That keeps every tensor offset valid.

Usage:
  python expert_repack.py --model in.gguf --stats stats.bin --out out.gguf [--verify] [--sha256]

The permutation is also written to <out>.perm.json. It maps output slot j to source expert perm[j].
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from gguf import GGUFReader  # noqa: E402

MAGIC = 0x53454F4D


def load_stats(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, n_layer, n_expert = struct.unpack_from("<IIII", data, 0)
    if magic != MAGIC or version != 1:
        raise SystemExit(f"unexpected stats header: magic=0x{magic:08x} version={version}")
    n = n_layer * n_expert
    if len(data) < 16 + n * 4:
        raise SystemExit("stats file is truncated")
    weights = struct.unpack_from(f"<{n}f", data, 16)
    return n_layer, n_expert, list(weights), data


def build_perm(n_layer, n_expert, weights):
    perm = []
    for il in range(n_layer):
        row = weights[il * n_expert:(il + 1) * n_expert]
        if sum(row) <= 1e-6:
            perm.append(list(range(n_expert)))
        else:
            perm.append(sorted(range(n_expert), key=lambda e: (-row[e], e)))
    return perm


def sha256_file(path, chunk=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tensor_is_expert(name):
    return name.endswith("_exps.weight") or name.endswith("_exps.bias")


def tensor_is_router(name):
    return name.endswith("ffn_gate_inp.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sha256", action="store_true")
    args = ap.parse_args()

    if os.path.abspath(args.model) == os.path.abspath(args.out):
        raise SystemExit("refusing to overwrite the input model")

    r = GGUFReader(args.model, "r")
    arch = r.fields["general.architecture"].contents()
    n_layer = int(r.fields["{}.block_count".format(arch)].contents())
    n_expert = int(r.fields["{}.expert_count".format(arch)].contents())

    s_layer, s_expert, weights, stats_raw = load_stats(args.stats)
    if s_layer != n_layer or s_expert != n_expert:
        raise SystemExit(f"stats mismatch: stats {s_layer}x{s_expert}, model {n_layer}x{n_expert}")
    perm = build_perm(n_layer, n_expert, weights)
    permuted_layers = sum(1 for p in perm if p != list(range(n_expert)))
    print(f"model {os.path.basename(args.model)}: {n_layer} layers, {n_expert} experts, "
          f"{permuted_layers} layers get a non-identity permutation")

    tensors = sorted(r.tensors, key=lambda t: t.data_offset)
    data_offset = r.data_offset
    file_size = os.path.getsize(args.model)

    n_transformed = 0
    with open(args.model, "rb") as src, open(args.out, "wb") as dst:
        dst.write(np.asarray(r.data[:data_offset]).tobytes())
        pos = data_offset
        for t in tensors:
            if t.data_offset < pos:
                raise SystemExit(f"overlapping tensor data at {t.name}")
            # padding between tensors is copied verbatim
            dst.write(np.asarray(r.data[pos:t.data_offset]).tobytes())
            raw = np.asarray(r.data[t.data_offset:t.data_offset + t.n_bytes])
            name = t.name
            if tensor_is_expert(name) and raw.size % n_expert == 0 and raw.size == t.n_bytes:
                per = raw.size // n_expert
                view = raw.reshape(n_expert, per)
                layer = None
                if name.startswith("blk."):
                    layer = int(name.split(".")[1])
                if layer is not None and 0 <= layer < n_layer:
                    dst.write(view[perm[layer]].tobytes())
                    n_transformed += 1
                else:
                    dst.write(raw.tobytes())
            elif tensor_is_router(name):
                layer = int(name.split(".")[1]) if name.startswith("blk.") else -1
                view = raw.view(np.float32) if raw.dtype != np.float32 else raw
                view = view.reshape(n_expert, view.size // n_expert)
                if 0 <= layer < n_layer:
                    dst.write(view[perm[layer]].tobytes())
                    n_transformed += 1
                else:
                    dst.write(raw.tobytes())
            else:
                dst.write(raw.tobytes())
            pos = t.data_offset + t.n_bytes
        dst.write(np.asarray(r.data[pos:file_size]).tobytes())

    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes), {n_transformed} tensors permuted")

    meta = {
        "model": os.path.abspath(args.model),
        "stats": os.path.abspath(args.stats),
        "stats_sha256": hashlib.sha256(stats_raw).hexdigest(),
        "out": os.path.abspath(args.out),
        "n_layer": n_layer,
        "n_expert": n_expert,
        "permuted_layers": permuted_layers,
        "perm": perm,
    }
    if args.sha256:
        meta["model_sha256"] = sha256_file(args.model)
        meta["out_sha256"] = sha256_file(args.out)
        print(f"model sha256 {meta['model_sha256']}")
        print(f"out   sha256 {meta['out_sha256']}")
    with open(args.out + ".perm.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"permutation written to {args.out}.perm.json")

    if args.verify:
        verify(args.model, args.out, perm, n_expert)


def verify(model_path, out_path, perm, n_expert):
    a = GGUFReader(model_path, "r")
    b = GGUFReader(out_path, "r")
    if a.data_offset != b.data_offset:
        raise SystemExit("verify: data offsets differ")
    with open(model_path, "rb") as fa, open(out_path, "rb") as fb:
        if fa.read(a.data_offset) != fb.read(b.data_offset):
            raise SystemExit("verify: header/KV/tensor-info section differs")

    by_name_a = {t.name: t for t in a.tensors}
    by_name_b = {t.name: t for t in b.tensors}
    if set(by_name_a) != set(by_name_b):
        raise SystemExit("verify: tensor sets differ")

    checked = 0
    for name, ta in by_name_a.items():
        tb = by_name_b[name]
        if ta.n_bytes != tb.n_bytes:
            raise SystemExit(f"verify: size differs for {name}")
        if tensor_is_expert(name):
            layer = int(name.split(".")[1])
            per = ta.n_bytes // n_expert
            src = np.asarray(ta.data).reshape(n_expert, per)
            dst = np.asarray(tb.data).reshape(n_expert, per)
            if not np.array_equal(dst, src[perm[layer]]):
                raise SystemExit(f"verify: payload differs for {name}")
            checked += 1
        elif tensor_is_router(name):
            layer = int(name.split(".")[1])
            src = np.asarray(ta.data).reshape(n_expert, -1)
            dst = np.asarray(tb.data).reshape(n_expert, -1)
            if not np.array_equal(dst, src[perm[layer]]):
                raise SystemExit(f"verify: payload differs for {name}")
            checked += 1
    print(f"verify: header identical, {checked} permuted tensors match the permutation")


if __name__ == "__main__":
    main()
