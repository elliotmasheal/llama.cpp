"""Reads GGUF metadata and tensor sizes only (no weight data is loaded).

Usage: python gguf_dims.py <model.gguf> [more.gguf ...]
Uses the repo's gguf-py. Prints the MoE dimensions cited in FEASIBILITY.md.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from gguf import GGUFReader  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB
# output projection GEMV measured at 217 GB/s, bench/RESULTS.md section 5
GPU_BW_MEASURED = 217e9


def field(r, key, default=None):
    f = r.fields.get(key)
    return default if f is None else f.contents()


def main(paths):
    for p in paths:
        r = GGUFReader(p, "r")
        arch = field(r, "general.architecture")
        n_block = field(r, f"{arch}.block_count")
        n_expert = field(r, f"{arch}.expert_count")
        top_k = field(r, f"{arch}.expert_used_count")
        n_embd = field(r, f"{arch}.embedding_length")
        ff = field(r, f"{arch}.expert_feed_forward_length")
        nextn = field(r, f"{arch}.nextn_predict_layers", 0)

        exp_bytes = 0
        exp_layers = set()
        dense_bytes = 0
        emb_bytes = 0
        for t in r.tensors:
            nb = int(t.n_bytes)
            if "_exps" in t.name:
                exp_bytes += nb
                exp_layers.add(t.name.split(".")[1])
            elif t.name == "token_embd.weight":
                emb_bytes += nb
            else:
                dense_bytes += nb

        n_moe = len(exp_layers)
        n_main = n_block - nextn
        per_expert = exp_bytes / (n_moe * n_expert)
        per_token_layers = min(n_moe, n_main)
        per_token = top_k * per_token_layers * per_expert
        dense_floor_ms = dense_bytes / GPU_BW_MEASURED * 1e3

        print(f"== {os.path.basename(p)} arch={arch}")
        print(f"  block_count={n_block} nextn={nextn} moe_layers_in_file={n_moe} "
              f"expert_count={n_expert} top_k={top_k} n_embd={n_embd} expert_ff={ff}")
        print(f"  expert bytes total = {exp_bytes / GIB:.3f} GiB")
        print(f"  per expert-layer (gate+up+down) = {per_expert / MIB:.3f} MiB")
        print(f"  expert bytes read per token (top_k x {per_token_layers} layers) = {per_token / MIB:.1f} MiB")
        print(f"  dense non-expert bytes (incl. output head, excl. token_embd) = {dense_bytes / GIB:.3f} GiB")
        print(f"  token_embd bytes = {emb_bytes / GIB:.3f} GiB")
        print(f"  dense read floor at 217 GB/s = {dense_floor_ms:.2f} ms/token")


if __name__ == "__main__":
    main(sys.argv[1:])
