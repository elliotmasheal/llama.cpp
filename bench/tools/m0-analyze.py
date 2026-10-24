"""M1 analyzer: turn a routing trace into a decode period / TG estimate.

Two numbers per candidate verify width k come from the trace:
  U_k   mean unique experts per layer over k consecutive decode tokens
  miss  expert misses for an LRU cache of C entries warmed on the prefill

Time model (single stream, all numbers from bench/RESULTS.md unless noted):
  dense per pass   2.17 GB  read once per pass regardless of k
  expert per pass  U_k * 40 layers * 1.93 MB (1.84 MiB per expert)
  VRAM bandwidth   217 GB/s
  H2D bandwidth    14.3 GB/s
  period = max(compute, h2d) + draft cost (restricted vocab, ~0.15 ms/step)
"""
import struct, sys
from collections import OrderedDict

DENSE_GB   = 2.17
EXPERT_MB  = 1.93
N_LAYER    = 40
VRAM_GBPS  = 217.0
H2D_GBPS   = 14.3
DRAFT_MS   = 0.15

path, prefill = sys.argv[1], int(sys.argv[2])
cache_fracs = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.20', '0.25', '0.30'])]
# measured acceptance per cycle for full-vocab drafts; restricted vocab is assumed equal here
E_BY_K = {3: 2.84, 5: 3.24, 9: 3.82}

with open(path, 'rb') as f:
    n_tok, n_layer, K = struct.unpack('<3i', f.read(12))
    lay = []
    for _ in range(n_layer):
        n_ints, = struct.unpack('<i', f.read(4))
        lay.append(struct.unpack('<%di' % n_ints, f.read(4 * n_ints)))

lens = [len(x) // K for x in lay]
n_dec = n_tok - prefill
# layer 39 keeps only the rows the last layer needs, so its prefill record is partial; its
# decode records are still the last n_dec entries, which lets all 40 layers enter decode stats.
use = [l for l in range(n_layer) if lens[l] >= prefill]     # complete prefill record
dec = [l for l in range(n_layer) if lens[l] >= n_dec]       # has the full decode tail
off = {l: lens[l] - n_dec for l in dec}
print('trace %s' % path)
print('tokens=%d prefill=%d decode=%d' % (n_tok, prefill, n_dec))
print('layers with complete prefill: %d, with full decode tail: %d' % (len(use), len(dec)))
print('prefill-record only layers: %s' % [l for l in range(n_layer) if l not in use])
print('decode-record only layers: %s' % [l for l in range(n_layer) if l not in dec])

def route(l, t):
    """row t counted from the start of the file (used for prefill-only layers)."""
    return lay[l][t * K:(t + 1) * K][:K]

def droote(l, i):
    """decode row i, counted from the start of the decode tail."""
    return lay[l][(off[l] + i) * K:(off[l] + i + 1) * K][:K]

def unique_per_layer(k):
    vals = []
    for s in range(0, n_dec - k + 1, k):
        acc = 0
        for l in dec:
            u = set()
            for i in range(s, s + k):
                u.update(droote(l, i))
            acc += len(u)
        vals.append(acc / len(dec))
    return sum(vals) / len(vals) if vals else float('nan')

def lru_miss_per_window(cap, k):
    """prefill-warmed LRU over all usable layers; return misses per k-token decode window."""
    c = OrderedDict()
    for t in range(prefill):
        for l in use:
            for e in route(l, t):
                key = (l, e)
                if key in c:
                    c.move_to_end(key)
                else:
                    c[key] = True
                    if len(c) > cap:
                        c.popitem(last=False)
    misses = windows = 0
    for s in range(0, n_dec - k + 1, k):
        for i in range(s, s + k):
            for l in dec:
                for e in droote(l, i):
                    key = (l, e)
                    if key in c:
                        c.move_to_end(key)
                    else:
                        misses += 1
                        c[key] = True
                        if len(c) > cap:
                            c.popitem(last=False)
        windows += 1
    return misses / windows if windows else float('nan')

TOTAL = 256 * len(use)
CPU_MS_PER_EXPERT = 0.108   # measured, 4 threads, 8 experts per layer (bench/RESULTS.md)
for frac in cache_fracs:
    cap = int(TOTAL * frac)
    print('\n[cache %.0f%% = %d experts = %.2f GiB]' % (frac * 100, cap, cap * EXPERT_MB / 1024))
    print('  k   U_k   miss/win  gpu_ms  h2d_ms  cpu_ms  E     TG(h2d)  TG(cpu)')
    for k in (2, 3, 4, 5, 6, 8, 9):
        u = unique_per_layer(k)
        miss = lru_miss_per_window(cap, k)
        hit_per_layer = u - miss / len(use)
        gpu = (DENSE_GB + hit_per_layer * len(use) * EXPERT_MB / 1024) / VRAM_GBPS * 1000
        h2d = miss * EXPERT_MB / 1024 / H2D_GBPS * 1000
        cpu = miss * CPU_MS_PER_EXPERT
        draft = (k - 1) * DRAFT_MS
        period_h2d = max(gpu, h2d) + draft
        period_cpu = max(gpu, cpu) + draft
        e = E_BY_K.get(k, float('nan'))
        def tg(p):
            return e / p * 1000 if (e == e and p > 0) else float('nan')
        print('  %d  %5.1f  %8.1f  %6.1f  %6.1f  %6.1f  %5.2f  %7.1f  %7.1f' % (
            k, u, miss, gpu, h2d, cpu, e, tg(period_h2d), tg(period_cpu)))
