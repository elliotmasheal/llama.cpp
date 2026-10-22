import re
lines = open(r"bench\expert-tiering\pp-prof.err", encoding="utf-8", errors="ignore").read().splitlines()
opre = re.compile(r"^([A-Z_0-9]+)(?: (.+?))?: (\d+) x ([\d.]+) us = ([\deE+.]+) us")
tables, cur = [], []
for line in lines:
    m = opre.match(line)
    if m:
        cur.append((m.group(1), (m.group(2) or ""), float(m.group(4)), float(m.group(5))))
        continue
    m = re.match(r"^Total time: ([\deE+.]+) us", line)
    if m:
        tables.append((cur, float(m.group(1)))); cur = []
half = len(tables)//2
for gi, (ops, tt) in enumerate(tables[:half]):
    s = sum(t for *_, t in ops)
    assert abs(s-tt) < 1.0, (gi, s, tt)
print("tables:", len(tables), "all tables sum to their Total time")
# aggregate per op (+shape) over eval 0
agg = {}
for name, sub, t, total in [(o[0], o[1], o[2], o[3]) for ops,_ in tables[:half] for o in ops]:
    key = (name, sub)
    if key in agg: agg[key] += total
    else: agg[key] = total
for (name, sub), v in sorted(agg.items(), key=lambda kv: -kv[1])[:20]:
    print(f"{v/1000:8.2f} ms  {name} {sub}")
print("sum of eval0 kernel time: %.1f ms" % (sum(agg.values())/1000))
