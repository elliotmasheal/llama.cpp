#!/usr/bin/env python3
"""Attention and dense-part accounting for the qwen36 prefill PP model.

Reads the same tensor table `qrt --gguf-index <model>` prints and derives, without a
model, the per-token FLOPs and per-layer weight bytes of the two attention layer types
(DeltaNet and full attention) plus the shared expert and the router. The expert (routed)
leg is measured elsewhere (RESULTS sections 39-44); this tool only covers what is left.

Usage:
    qrt --gguf-index model.gguf > tensors.csv
    python bench/tools/attn_account.py tensors.csv [--tokens 10240]

Every number is a shape product, not a measurement: it exists so the PP model has a
traceable denominator and so a missing tensor shows up as an error instead of a guess.
"""

import argparse
import collections
import sys


def rows(path):
    out = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        f = line.split(',')
        if len(f) < 10 or f[1].isdigit() is False:
            continue
        out.append({
            'name': f[0],
            'dims': [int(f[2]), int(f[3]), int(f[4]), int(f[5])],
            'type': f[7],
            'bytes': int(f[9]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--tokens', type=int, default=10240, help='prompt tokens')
    ap.add_argument('--dense-tflops', type=float, default=6.0,
                    help='assumed dense GEMM rate (no Q8_0 kernel yet)')
    ap.add_argument('--attn-tflops', type=float, default=6.0)
    ap.add_argument('--expert-tflops', type=float, default=7.1,
                    help='measured expert leg rate (RESULTS 44)')
    args = ap.parse_args()

    t = {}
    for r in rows(args.csv):
        t[r['name']] = r

    def need(name):
        if name not in t:
            sys.exit('missing tensor %s' % name)
        return t[name]

    # Full attention layer: q carries query+gate, k/v are GQA, head_dim from the norms.
    head_dim = need('blk.3.attn_q_norm.weight')['dims'][0]
    q_out = need('blk.3.attn_q.weight')['dims'][1]
    n_head = q_out // (2 * head_dim)
    kv_out = need('blk.3.attn_k.weight')['dims'][1]
    n_kv = kv_out // head_dim
    o_in = need('blk.3.attn_output.weight')['dims'][0]
    n_embd = need('blk.3.attn_norm.weight')['dims'][0]

    # DeltaNet layer: qkv packs q and k (both head_k_dim x ssm_n_group) plus v.
    qkv_out = need('blk.0.attn_qkv.weight')['dims'][1]
    z_out = need('blk.0.attn_gate.weight')['dims'][1]
    n_v_heads = need('blk.0.ssm_alpha.weight')['dims'][1]
    head_v_dim = need('blk.0.ssm_norm.weight')['dims'][0]
    conv_chan = need('blk.0.ssm_conv1d.weight')['dims'][1]
    conv_k = need('blk.0.ssm_conv1d.weight')['dims'][0]
    v_dim = n_v_heads * head_v_dim
    qk_dim = (qkv_out - v_dim) // 2
    n_k_heads = qk_dim // head_v_dim
    head_k_dim = head_v_dim

    layers = collections.defaultdict(list)
    for r in rows(args.csv):
        f = r['name'].split('.')
        if len(f) >= 3 and f[0] == 'blk':
            layers[int(f[1])].append(r['name'])
    dn = [l for l, ns in layers.items() if any(n.endswith('ssm_a') for n in ns)]
    fa = [l for l, ns in layers.items() if any(n.endswith('attn_k.weight') for n in ns)]

    def sum_bytes(names):
        return sum(t[n]['bytes'] for n in names if n in t)

    def dense_of(l):
        return [n for n in layers[l] if '_exps' not in n]

    dn_bytes = sum(sum_bytes(dense_of(l)) for l in dn)
    fa_bytes = sum(sum_bytes(dense_of(l)) for l in fa)

    # FLOPs: every dense matmul is 2 x out x in per token. The attention and recurrence
    # terms are per token or per token pair as noted.
    def mm(a, b):
        return 2 * a * b

    def lora(name):
        r = need(name)
        return mm(r['dims'][0], r['dims'][1])

    # DeltaNet per token
    dn_rank = need('blk.0.ssm_alpha.weight')['dims'][1]
    dn_flops = (mm(n_embd, qkv_out) + mm(n_embd, z_out) + mm(v_dim, n_embd) +
                lora('blk.0.ssm_alpha.weight') + lora('blk.0.ssm_beta.weight') +
                2 * conv_chan * conv_k +
                # delta rule: state [head_k, head_v] per head, three passes (kv, k-state, outer)
                3 * 2 * head_k_dim * head_v_dim * n_v_heads +
                mm(n_embd, need('blk.0.ffn_gate_inp.weight')['dims'][1]) +
                3 * mm(n_embd, need('blk.0.ffn_gate_shexp.weight')['dims'][1]))

    # full attention per token, projections only (the q x k term is quadratic, below)
    fa_flops = (mm(n_embd, q_out) + 2 * mm(n_embd, kv_out) + mm(o_in, n_embd) +
                mm(n_embd, need('blk.3.ffn_gate_inp.weight')['dims'][1]) +
                3 * mm(n_embd, need('blk.3.ffn_gate_shexp.weight')['dims'][1]))

    # attention proper over the prompt: each token attends to half the prompt on average
    attn_pairs = args.tokens * (args.tokens + 1) / 2.0
    fa_attn = attn_pairs * n_head * 2 * (2 * head_dim)   # QK^T and PV, 2 flops per mac

    n_all = len(dn) + len(fa)
    dense_tflops = (len(dn) * dn_flops + len(fa) * fa_flops) * args.tokens / 1e12
    attn_tflops = len(fa) * fa_attn / 1e12
    expert_tflops = 21.1312   # measured: 2.06 GFLOP/token x 10240, RESULTS 44

    print('shapes')
    print('  n_embd=%d head_dim=%d n_head=%d n_kv=%d (full attention)' % (n_embd, head_dim, n_head, n_kv))
    print('  delta net: head_k=%d head_v=%d n_k_heads=%d n_v_heads=%d conv1d k=%d channels=%d'
          % (head_k_dim, head_v_dim, n_k_heads, n_v_heads, conv_k, conv_chan))
    print('  layers: %d total, %d delta net, %d full attention' % (n_all, len(dn), len(fa)))
    print('  dense weights: delta %.1f MiB, full %.1f MiB, total %.0f MiB'
          % (dn_bytes / 2**20, fa_bytes / 2**20, (dn_bytes + fa_bytes) / 2**20))
    print()
    print('per token FLOPs (dense part, expert leg excluded)')
    print('  delta net layer   %8.1f MFLOP' % (dn_flops / 1e6))
    print('  full attention    %8.1f MFLOP (+ attention proper, quadratic)' % (fa_flops / 1e6))
    print()
    print('prompt of %d tokens' % args.tokens)
    print('  dense projections %8.2f TFLOP' % dense_tflops)
    print('  attention proper  %8.2f TFLOP' % attn_tflops)
    print('  expert leg        %8.2f TFLOP (measured bytes/flops, RESULTS 44)' % expert_tflops)
    print('  total             %8.2f TFLOP' % (dense_tflops + attn_tflops + expert_tflops))
    print()
    print('PP model (assumed rates: dense %.1f, attention %.1f, expert %.1f TFLOPS)'
          % (args.dense_tflops, args.attn_tflops, args.expert_tflops))
    dense_s = (dense_tflops + attn_tflops) / args.dense_tflops
    expert_s = expert_tflops / args.expert_tflops
    print('  dense+attention   %8.2f s' % dense_s)
    print('  expert            %8.2f s (measured 2.976 s incl. transport)' % expert_s)
    print('  sum               %8.2f s -> %.0f tok/s' % (dense_s + expert_s, args.tokens / (dense_s + expert_s)))
    print('  dense share of FLOPs: %.0f%%' % (100.0 * (dense_tflops + attn_tflops) /
                                              (dense_tflops + attn_tflops + expert_tflops)))


if __name__ == '__main__':
    main()
