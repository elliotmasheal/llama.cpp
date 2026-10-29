# Offline analysis of a LLAMA_MOE_TRACE file: how much of the decode hit rate is the policy's fault?
# Record format (src/llama-moe-cache.cpp: trace_ids): int32 layer, int32 count, count x int32 expert id.
# One record per (ubatch, layer); a ubatch boundary is the layer index going backwards.
import argparse
import struct
import sys
from collections import OrderedDict


def read_records(path):
    with open(path, "rb") as f:
        data = f.read()
    out = []
    off = 0
    n = len(data)
    while off + 8 <= n:
        il, cnt = struct.unpack_from("<ii", data, off)
        off += 8
        if cnt < 0 or off + 4 * cnt > n:
            raise ValueError("truncated record at %d" % off)
        ids = struct.unpack_from("<%di" % cnt, data, off)
        off += 4 * cnt
        out.append((il, ids))
    return out


def split_ubatches(records):
    # a new ubatch starts when the layer index does not increase
    batches = []
    cur = []
    last = -1
    for il, ids in records:
        if il <= last:
            batches.append(cur)
            cur = []
        cur.append((il, ids))
        last = il
    if cur:
        batches.append(cur)
    return batches


def decode_batches(batches, n_tokens, n_layers):
    # a decode batch carries n_tokens ids per layer; keep the trailing single-token ones
    out = []
    for b in batches:
        if not b:
            continue
        cnt = len(b[0][1])
        if len(b) == n_layers and cnt <= 8 * n_tokens:
            out.append(b)
    return out


def lru_hits(seq, capacity):
    # seq: list of (layer, expert) in time order; capacity counts expert entries for the whole group
    lru = OrderedDict()
    hits = misses = 0
    for key in seq:
        if key in lru:
            hits += 1
            lru.move_to_end(key)
        else:
            misses += 1
            if len(lru) >= capacity:
                lru.popitem(last=False)
            lru[key] = 1
    return hits, misses


def static_hits(seq, capacity):
    freq = {}
    for key in seq:
        freq[key] = freq.get(key, 0) + 1
    top = set(sorted(freq, key=lambda k: -freq[k])[:capacity])
    hits = sum(1 for k in seq if k in top)
    return hits, len(seq) - hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--capacity", type=int, default=631, help="slots per group")
    ap.add_argument("--groups", default="0-19,20-39", help="layer ranges per group")
    args = ap.parse_args()

    records = read_records(args.trace)
    print("records: %d" % len(records))
    batches = split_ubatches(records)
    print("ubatches: %d" % len(batches))
    rows = [(len(b), len(b[0][1])) for b in batches]
    dec = [b for b in batches if len(b) == args.n_layers and len(b[0][1]) <= 8]
    print("looks like decode ubatches: %d" % len(dec))
    if not dec:
        return 1

    groups = []
    for part in args.groups.split(","):
        a, b = part.split("-")
        groups.append(set(range(int(a), int(b) + 1)))

    total_hits = total = 0
    for gi, glayers in enumerate(groups):
        seq = []
        for b in dec:
            for il, ids in b:
                if il in glayers:
                    seq.extend((il, int(e)) for e in ids)
        cap = args.capacity
        h_lru, m_lru = lru_hits(seq, cap)
        h_st, m_st = static_hits(seq, cap)
        total_hits += h_lru
        total += h_lru + m_lru
        print("group %d: selections %d, LRU(%d) hit %.2f%%, static top-%d hit %.2f%%"
              % (gi, len(seq), cap, 100.0 * h_lru / len(seq), cap, 100.0 * h_st / len(seq)))
        for c in (cap // 2, cap, cap * 3 // 2):
            h, m = lru_hits(seq, c)
            print("    LRU capacity %5d -> %.2f%%" % (c, 100.0 * h / (h + m)))
    print("overall LRU hit %.2f%%" % (100.0 * total_hits / total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
