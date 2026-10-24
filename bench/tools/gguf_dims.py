"""Reads GGUF metadata and tensor sizes only (no weight data is loaded).

Usage: python gguf_dims.py <model.gguf> [more.gguf ...]
Uses the repo's gguf-py. Prints the tensor account and the 256K KV budget.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
# gguf-py imports yaml for model-card frontmatter, which this tool never reads. Avoid the dependency.
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
from gguf import GGUFReader  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB
# bench/RESULTS.md section 8: large-buffer VRAM read ceiling, replaces the 217 GB/s figure.
GPU_BW_MEASURED = 156.5e9
# QRT_KV_PRECISION default is f16; design 7 forbids lowering it without a quality gate.
KV_BYTES_PER_ELEM = 2
VRAM_TOTAL_MIB = 8176
DESKTOP_MIB = 1000
COMPUTE_MIB = (700, 1100)
DRAFT_MIB = 100


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
        full_layers = set()
        for t in r.tensors:
            nb = int(t.n_bytes)
            parts = t.name.split(".")
            if len(parts) >= 3 and parts[0] == "blk" and parts[2] == "attn_k":
                full_layers.add(int(parts[1]))
            if "_exps" in t.name:
                exp_bytes += nb
                exp_layers.add(parts[1])
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
        print(f"  dense read floor at {GPU_BW_MEASURED / 1e9:.1f} GB/s = {dense_floor_ms:.2f} ms/token")

        n_kv = field(r, f"{arch}.attention.head_count_kv")
        k_len = field(r, f"{arch}.attention.key_length")
        v_len = field(r, f"{arch}.attention.value_length")
        n_ctx = field(r, f"{arch}.context_length")
        interval = field(r, f"{arch}.full_attention_interval")
        main_full = sorted(L for L in full_layers if L < n_main)
        mtp_full = sorted(L for L in full_layers if L >= n_main)
        kv_layer = n_kv * (k_len + v_len) * KV_BYTES_PER_ELEM

        print(f"  full_attention_interval={interval} n_ctx={n_ctx}")
        print(f"  full-attn layers: main={main_full} ({len(main_full)}), "
              f"nextn={mtp_full} ({len(mtp_full)})")
        print(f"  kv bytes/token/layer = {kv_layer} B "
              f"({n_kv} kv heads x ({k_len}+{v_len}) x {KV_BYTES_PER_ELEM} B)")
        print(f"  kv bytes/token main = {kv_layer * len(main_full)} B, "
              f"with nextn = {kv_layer * len(full_layers)} B")

        usable = VRAM_TOTAL_MIB - DESKTOP_MIB
        print(f"  vram budget: total={VRAM_TOTAL_MIB} MiB usable={usable} MiB "
              f"dense={dense_bytes / MIB:.0f} MiB compute={COMPUTE_MIB[0]}-{COMPUTE_MIB[1]} "
              f"draft={DRAFT_MIB} MiB")
        print("  ctx        kv_mib   cache_left_mib  cache_pct")
        for ctx in (10240, 16384, 32768, 65536, 131072, n_ctx):
            kv_mib = kv_layer * len(full_layers) * ctx / MIB
            left = usable - dense_bytes / MIB - COMPUTE_MIB[0] - DRAFT_MIB - kv_mib
            pct = left * MIB / exp_bytes * 100.0
            print(f"  {ctx:6d}  {kv_mib:9.1f}  {left:13.1f}  {pct:8.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
