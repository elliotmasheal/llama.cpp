"""Reads the LLAMA_MOE_TRACE binary trace and reports what repack / eviction design needs.

Record layout (little endian, native): int32 layer, int32 n, n x int32 expert id.
One record per traced MoE-cache lookup, in graph order, so layer ids wrap at each token.

Usage: python read_trace.py <trace.bin> [--top N]
"""
import array
import sys
from collections import Counter, defaultdict


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    a = array.array("i")
    a.frombytes(data[: len(data) // a.itemsize * a.itemsize])
    return a


def split_tokens(recs):
    """Groups records into tokens: a token ends when the layer id does not increase."""
    tokens = []
    cur = []
    last = None
    for il, ids in recs:
        if last is not None and il <= last:
            tokens.append(cur)
            cur = []
        cur.append((il, ids))
        last = il
    if cur:
        tokens.append(cur)
    return tokens


def main():
    path = sys.argv[1]
    top_n = 8
    if "--top" in sys.argv:
        top_n = int(sys.argv[sys.argv.index("--top") + 1])

    a = load(path)
    recs = []
    i = 0
    while i + 2 <= len(a):
        il, n = a[i], a[i + 1]
        i += 2
        if n < 0 or i + n > len(a):
            print(f"truncated record at {i}")
            break
        recs.append((il, list(a[i:i + n])))
        i += n

    tokens = [t for t in split_tokens(recs) if t]
    print(f"file: {path}")
    print(f"records: {len(recs)}, tokens: {len(tokens)}")
    if not tokens:
        return
    layers = max(il for il, _ in recs) + 1
    print(f"layers seen: {layers}")

    per_layer = defaultdict(Counter)
    pair = Counter()
    for tok in tokens:
        for il, ids in tok:
            per_layer[il].update(ids)
        for il, ids in tok:
            s = sorted(set(ids))
            for x in range(len(s)):
                for y in range(x + 1, len(s)):
                    pair[(il, s[x], s[y])] += 1

    total = len(tokens)
    print("\nper-layer top experts (share of tokens that selected them, and top-k coverage):")
    for il in sorted(per_layer):
        c = per_layer[il]
        tot = sum(c.values())
        ranked = c.most_common()
        top_share = sum(v for _, v in ranked[:top_n]) / tot if tot else 0.0
        top_tokens = [v / total for _, v in ranked[:8]]
        gini_num = sum((2 * (i + 1) - len(ranked) - 1) * v for i, (_, v) in enumerate(sorted(ranked, key=lambda x: x[1])))
        gini = gini_num / (len(ranked) * tot) if tot else 0.0
        print(f"  layer {il:2d}: distinct={len(ranked):3d} top{top_n}_share={top_share:.3f} gini={gini:.3f} "
              f"top8={[round(x, 3) for x in top_tokens]}")

    print("\nmost co-selected expert pairs per layer (top 5):")
    for il in sorted(per_layer):
        top = [(k, v) for k, v in pair.most_common() if k[0] == il][:5]
        if top:
            print(f"  layer {il:2d}: " + ", ".join(f"({a_},{b_})x{c_}" for (_, a_, b_), c_ in top))


if __name__ == "__main__":
    main()
