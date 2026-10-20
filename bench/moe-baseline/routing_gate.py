#!/usr/bin/env python3
# routing predictability gate: parse the LLAMA_ROUTING_DIAG dumps and report recall / miss bytes.
# binary format per ubatch: int32 n_tok, int32 n_lay, then per layer: int32 il, n_expert, n_sel,
# n_keep, int64 expert_bytes, then n_tok x n_keep int32 (top-(n_sel+32) expert ids, prob desc).
import struct
import os

H2D_GBPS = 13.9          # measured pinned host->device ceiling
EXPERT_TRIPLE_MIB = 1.945  # gate+up+down expert slice, approximate (5 layers use q4_K down)
EB = EXPERT_TRIPLE_MIB * 1024 * 1024


def parse(path):
    data = open(path, 'rb').read()
    off = 0
    ubs = []
    hdr = struct.Struct('<ii')
    lay = struct.Struct('<iiiiq')
    while off < len(data):
        n_tok, n_lay = hdr.unpack_from(data, off)
        off += hdr.size
        layers = []
        for _ in range(n_lay):
            il, n_exp, n_sel, n_keep, ebytes = lay.unpack_from(data, off)
            off += lay.size
            rows = []
            for _t in range(n_tok):
                rows.append(struct.unpack_from('<%di' % n_keep, data, off))
                off += 4 * n_keep
            layers.append({'il': il, 'n_exp': n_exp, 'n_sel': n_sel, 'n_keep': n_keep, 'rows': rows})
        ubs.append({'n_tok': n_tok, 'layers': layers})
    assert off == len(data)
    return ubs


def analyse(path, label):
    ubs = parse(path)
    Ms = (8, 16, 32)
    st = {M: {'pairs': 0, 'hits': 0, 'misses': 0, 'incr': 0, 'tok_mis': 0,
              'b_sup': 0.0, 'b_miss': 0.0, 'b_incr': 0.0} for M in Ms}
    n_pairs = 0
    for ub in ubs:
        n_tok = ub['n_tok']
        if n_tok < 2:
            continue
        n_pairs += n_tok - 1
        prev = {M: {} for M in Ms}
        # token-major so the per-token fallback rate can be computed across layers
        tok_any_mis = {M: 0 for M in Ms}
        for t in range(n_tok - 1):
            for M in Ms:
                any_mis = False
                for L in ub['layers']:
                    il = L['il']
                    sup = set(L['rows'][t][:8 + M])
                    actual = set(L['rows'][t + 1][:8])
                    hits = len(actual & sup)
                    p = prev[M].get(il)
                    incr = len(sup - p) if p is not None else len(sup)
                    prev[M][il] = sup
                    s = st[M]
                    s['pairs'] += 1
                    s['hits'] += hits
                    s['misses'] += 8 - hits
                    s['incr'] += incr
                    s['b_sup'] += len(sup) * EB
                    s['b_miss'] += (8 - hits) * EB
                    s['b_incr'] += incr * EB
                    if hits < 8:
                        any_mis = True
                if any_mis:
                    tok_any_mis[M] += 1
        for M in Ms:
            st[M]['tok_mis'] += tok_any_mis[M]
    print("== %s (%s): %d ubatches, %d token pairs per layer ==" %
          (label, os.path.basename(path), len(ubs), n_pairs))
    for M in Ms:
        s = st[M]
        pairs = max(1, s['pairs'])
        layers = max(1, len(ubs[0]['layers'])) if ubs else 1
        tok = max(1, s['pairs'] // layers)
        rec = 100.0 * s['hits'] / (8 * pairs)
        lay_mis = 100.0 * s['misses'] / (8 * pairs)
        tok_mis = 100.0 * s['tok_mis'] / tok
        b_sup = s['b_sup'] / tok / 1e6
        b_miss = s['b_miss'] / tok / 1e6
        b_incr = s['b_incr'] / tok / 1e6
        print("M=%2d superset=%2d  recall=%5.1f%%  layer_miss=%5.1f%%  token_fallback=%5.1f%%"
              % (M, 8 + M, rec, lay_mis, tok_mis))
        print("     bytes/token: superset %7.1f MB -> %6.1f t/s | miss %6.1f MB -> %7.1f t/s | incremental %7.1f MB -> %6.1f t/s"
              % (b_sup, 13900.0 / max(b_sup, 1e-9), b_miss, 13900.0 / max(b_miss, 1e-9),
                 b_incr, 13900.0 / max(b_incr, 1e-9)))


if __name__ == '__main__':
    for path, label in [(r'bench\moe-baseline\routing_514.bin', '514-token prompt'),
                        (r'bench\moe-baseline\routing_10k.bin', 'prompt10k')]:
        if os.path.exists(path):
            analyse(path, label)
        else:
            print('missing', path)
