import re
txt = open(r"bench\expert-tiering\pp-prof.err", encoding="utf-8", errors="ignore").read().splitlines()
opre = re.compile(r"^([A-Z_0-9]+) (.+?): (\d+) x ([\d.]+) us = ([\deE+.]+) us")
graphs, cur = [], []
for line in txt:
    m = opre.match(line)
    if m: cur.append((line.split(":")[0], float(m.group(5)))); continue
    m = re.match(r"^Total time: ([\deE+.]+) us", line)
    if m: graphs.append((cur, float(m.group(1)))); cur = []
half = len(graphs)//2
for gi in (3,4):
    ops, tt = graphs[gi]
    print(f"=== graph {gi} total {tt/1000:.1f} ms, {len(ops)} ops ===")
    for name, t in ops: print(f"  {t/1000:7.2f} ms  {name}")
