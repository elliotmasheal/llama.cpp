import re
txt = open(r"bench\expert-tiering\pp-prof.err", encoding="utf-8", errors="ignore").read().splitlines()
opre = re.compile(r"^([A-Z_0-9]+) (.+?): (\d+) x ([\d.]+) us = ([\deE+.]+) us")
graphs = []
cur = []
for line in txt:
    m = opre.match(line)
    if m: cur.append((m.group(1), float(m.group(5)))); continue
    m = re.match(r"^Total time: ([\deE+.]+) us", line)
    if m: graphs.append((cur, float(m.group(1)))); cur = []
half = len(graphs)//2
gaps = [(tt/1000, sum(t for _,t in ops)/1000, len(ops)) for ops, tt in graphs[:half]]
print("eval0 graphs:", len(gaps))
for i,(tt,op,n) in enumerate(gaps):
    print(f"{i:2d} total {tt:7.1f} ops {op:7.1f} gap {tt-op:6.1f} nops {n}")
