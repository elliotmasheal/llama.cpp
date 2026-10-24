import struct, sys
from collections import OrderedDict

# File format (m0-trace.cpp): int32 n_tok, n_layer, k; then per layer: int32 n_ints, n_ints int32 ids.
path = sys.argv[1]
PREFILL = int(sys.argv[2])
with open(path, 'rb') as f:
    n_tok, n_layer, k = struct.unpack('<3i', f.read(12))
    lay = []
    for _ in range(n_layer):
        n_ints, = struct.unpack('<i', f.read(4))
        lay.append(struct.unpack('<%di' % n_ints, f.read(4 * n_ints)))

lens = [len(x) // k for x in lay]
print('n_tok=%d layers=%d k=%d prefill=%d' % (n_tok, n_layer, k, PREFILL))
print('per-layer traced tokens:', lens)

# Use layers with a complete prefill record; per-layer decode records may be missing (see RESULTS notes).
use = [l for l in range(n_layer) if lens[l] >= PREFILL]
dec_full = [l for l in use if lens[l] == n_tok]
print('layers usable for prefill stats: %d, layers with full decode record: %s' % (len(use), dec_full))

def route(l, t):
    return lay[l][t * k:(t + 1) * k]

TOTAL = 256 * len(use)

def run_lru(cap, tok_range, warm_range, layers):
    cache = OrderedDict()
    hits = total = 0
    for t in warm_range:
        for l in layers:
            for e in route(l, t):
                key = (l, e)
                if key in cache:
                    cache.move_to_end(key)
                else:
                    cache[key] = True
                    if len(cache) > cap:
                        cache.popitem(last=False)
    for t in tok_range:
        for l in layers:
            for e in route(l, t):
                key = (l, e)
                total += 1
                if key in cache:
                    hits += 1
                    cache.move_to_end(key)
                else:
                    cache[key] = True
                    if len(cache) > cap:
                        cache.popitem(last=False)
    return 100.0 * hits / max(total, 1)

def run_static(cap, fit_range, test_range, layers):
    freq = {}
    for t in fit_range:
        for l in layers:
            for e in route(l, t):
                freq[(l, e)] = freq.get((l, e), 0) + 1
    pinned = set(sorted(freq, key=lambda x: -freq[x])[:cap])
    hits = total = 0
    for t in test_range:
        for l in layers:
            for e in route(l, t):
                total += 1
                hits += (l, e) in pinned
    return 100.0 * hits / max(total, 1)

fracs = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50)
print('\n[A] global LRU (budget = fraction of %d experts in usable layers)' % TOTAL)
print('  prefill-warmed LRU, hit on decode tokens (only layers with full decode record):')
for fr in fracs:
    cap = int(TOTAL * fr)
    print('    cap=%5d (%4.1f%%, %.2f GiB): decode hit=%.1f%%' % (cap, fr * 100, cap * 1.83 / 1024,
          run_lru(cap, range(PREFILL, n_tok), range(0, PREFILL), dec_full)))
print('  cold LRU over prefill, hit on prefill tokens 2048..%d (all usable layers):' % (PREFILL - 1))
for fr in fracs:
    cap = int(TOTAL * fr)
    print('    cap=%5d (%4.1f%%): prefill hit=%.1f%%' % (cap, fr * 100,
          run_lru(cap, range(2048, PREFILL), range(0, 2048), use)))

print('\n[B] static top-N by prefill frequency, out-of-sample on decode (full-decode layers)')
for fr in (0.10, 0.20, 0.25, 0.30):
    cap = int(TOTAL * fr)
    print('    cap=%5d (%4.1f%%): decode hit=%.1f%%' % (cap, fr * 100,
          run_static(cap, range(0, PREFILL), range(PREFILL, n_tok), dec_full)))

print('\n[C] mean unique experts per layer over window of w consecutive decode tokens (full-decode layers)')
for w in (1, 2, 4, 8):
    vals = []
    for s in range(PREFILL, n_tok - w + 1, w):
        acc = []
        for l in dec_full:
            u = set()
            for t in range(s, s + w):
                u.update(route(l, t))
            acc.append(len(u))
        vals.append(sum(acc) / len(acc))
    if vals:
        print('    w=%d: unique/layer=%.1f  (random-routing expectation %.1f)' % (
            w, sum(vals) / len(vals), 256 * (1 - (1 - 8 / 256) ** w)))

print('\n[D] reuse of previous token expert at t-1 (full-decode layers)')
same = tot = 0
for t in range(PREFILL + 1, n_tok):
    for l in dec_full:
        prev = set(route(l, t - 1))
        for e in route(l, t):
            tot += 1
            same += e in prev
print('    reuse(t-1) = %.1f%%' % (100.0 * same / max(tot, 1)))
