# Offline ceiling and policy comparison for the expert bank at a fixed size, from a LLAMA_MOE_TRACE file.
# Answers "is there hit rate left without growing the bank": Belady gives the absolute ceiling, static top-N
# and pinned-per-layer plus shared LRU give what a real policy can reach, LFU shows why naive frequency fails.
#
#   python bench/tools/moe_trace_policy.py trace.bin --capacity 1264
import argparse
import sys
from collections import Counter, OrderedDict

from moe_trace_stats import read_records, split_ubatches

N_LAYERS = 40


def load_seq(path, n_layers):
    dec = [b for b in split_ubatches(read_records(path))
           if len(b) == n_layers and len(b[0][1]) <= 8]
    seq = []
    sets = []
    for b in dec:
        sets.append({il: set(int(e) for e in ids) for il, ids in b})
        for il, ids in b:
            seq.extend((il, int(e)) for e in ids)
    return dec, seq, sets


def belady_hits(seq, cap):
    n = len(seq)
    nxt = [n + 1]*n
    last = {}
    for i in range(n - 1, -1, -1):
        k = seq[i]
        nxt[i] = last.get(k, n + 1)
        last[k] = i
    cache = {}
    hits = 0
    for i, k in enumerate(seq):
        if k in cache:
            hits += 1
            cache[k] = nxt[i]
        else:
            if len(cache) >= cap:
                del cache[max(cache, key=cache.get)]
            cache[k] = nxt[i]
    return hits


def lru_hits(seq, cap):
    lru = OrderedDict()
    hits = 0
    for k in seq:
        if k in lru:
            hits += 1
            lru.move_to_end(k)
        else:
            if len(lru) >= cap:
                lru.popitem(last=False)
            lru[k] = 1
    return hits


def lfu_hits(seq, cap):
    cnt = Counter()
    cache = set()
    hits = 0
    for k in seq:
        cnt[k] += 1
        if k in cache:
            hits += 1
        else:
            if len(cache) >= cap:
                cache.remove(min(cache, key=lambda x: cnt[x]))
            cache.add(k)
    return hits


def static_hits(seq, cap):
    top = set(k for k, _ in Counter(seq).most_common(cap))
    return sum(1 for k in seq if k in top)


def pinned_plus_lru(seq, cap, k_per_layer, n_layers):
    freq = Counter(seq)
    pinned = set()
    for il in range(n_layers):
        cand = sorted((k for k in freq if k[0] == il), key=lambda x: -freq[x])
        pinned.update(cand[:k_per_layer])
    rest = cap - len(pinned)
    if rest <= 0:
        return sum(1 for k in seq if k in pinned)
    lru = OrderedDict()
    hits = 0
    for k in seq:
        if k in pinned:
            hits += 1
        elif k in lru:
            hits += 1
            lru.move_to_end(k)
        else:
            if len(lru) >= rest:
                lru.popitem(last=False)
            lru[k] = 1
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--capacity", type=int, default=1264)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    args = ap.parse_args()

    dec, seq, sets = load_seq(args.trace, args.n_layers)
    n = len(seq)
    cap = args.capacity
    print("decode tokens %d, selections %d, capacity %d" % (len(dec), n, cap))
    print("Belady OPT     %.2f%%  (ceiling, needs the future)" % (100.0*belady_hits(seq, cap)/n))
    print("static top-N   %.2f%%" % (100.0*static_hits(seq, cap)/n))
    print("LRU            %.2f%%" % (100.0*lru_hits(seq, cap)/n))
    print("LFU            %.2f%%  (aging-less frequency, evicts new entries)" % (100.0*lfu_hits(seq, cap)/n))
    for k in (8, 16, 24, 32):
        print("pin %2d/layer   %.2f%%" % (k, 100.0*pinned_plus_lru(seq, cap, k, args.n_layers)/n))

    rep = tot = 0
    for i in range(1, len(sets)):
        for il in sets[i]:
            rep += len(sets[i][il] & sets[i - 1][il])
            tot += len(sets[i][il])
    print("adjacent token repeat, same layer: %.2f%%" % (100.0*rep/max(1, tot)))
    per_layer = [len(set().union(*[s[il] for s in sets])) for il in sets[0]]
    per_layer.sort()
    print("distinct experts per layer over %d tokens: min %d median %d max %d"
          % (len(dec), per_layer[0], per_layer[len(per_layer)//2], per_layer[-1]))


if __name__ == "__main__":
    sys.exit(main())
