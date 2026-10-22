import re
txt = open(r"bench\expert-tiering\pp-prof.err", encoding="utf-8", errors="ignore").read().splitlines()
opre = re.compile(r"^([A-Z_0-9]+) (.+?): (\d+) x ([\d.]+) us = ([\deE+.]+) us")
graphs = []
cur_ops = []
cur_total = None
for line in txt:
    m = opre.match(line)
    if m:
        cur_ops.append((m.group(1), float(m.group(5))))
        continue
    m = re.match(r"^Total time: ([\deE+.]+) us", line)
    if m:
        graphs.append((cur_ops, float(m.group(1))))
        cur_ops = []
print("graphs:", len(graphs))
evals = [graphs[: len(graphs)//2], graphs[len(graphs)//2:]] if len(graphs) >= 2 and len(graphs) % 2 == 0 else [graphs]
for gi, g in enumerate(evals):
    tot_ops = sum(t for ops, _ in g for _, t in ops)
    tot_time = sum(t for _, t in g)
    print(f"eval {gi}: graphs={len(g)} opsum={tot_ops/1000:.1f} ms total_time_sum={tot_time/1000:.1f} ms")
    # per graph: ops vs total for the biggest few
for ops, tt in graphs[:4]:
    print(f"  graph total {tt/1000:.1f} ms, op sum {sum(t for _, t in ops)/1000:.1f} ms, nops {len(ops)}")
