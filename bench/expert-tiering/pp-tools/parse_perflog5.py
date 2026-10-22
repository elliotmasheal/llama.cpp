import re
lines = open(r"bench\expert-tiering\pp-prof.err", encoding="utf-8", errors="ignore").read().splitlines()
opre = re.compile(r"^([A-Z_0-9]+)(?: (.+?))?: (\d+) x ([\d.]+) us = ([\deE+.]+) us")
tables, cur, unmatched = [], [], []
for line in lines:
    m = opre.match(line)
    if m:
        cur.append((m.group(1), (m.group(2) or ""), float(m.group(5))))
        continue
    m = re.match(r"^Total time: ([\deE+.]+) us", line)
    if m:
        tables.append((cur, float(m.group(1)))); cur = []
        continue
    if re.match(r"^\S+.*: \d+ x [\d.]+ us", line) or (" x " in line and " us = " in line):
        unmatched.append(line)
half = len(tables)//2
agg = {}
for ops, tt in tables[:half]:
    for name, sub, total in ops:
        agg[(name, sub)] = agg.get((name, sub), 0) + total
print("unmatched lines:", len(unmatched))
for u in unmatched[:8]: print("  MISS:", u)
print("matched sum: %.1f ms" % (sum(agg.values())/1000))
for (name, sub), v in sorted(agg.items(), key=lambda kv: -kv[1])[:18]:
    print(f"{v/1000:8.2f} ms  {name} {sub}")
