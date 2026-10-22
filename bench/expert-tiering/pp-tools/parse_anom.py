import re
tot_h2d_us = 0; tot_bytes = 0; n = 0
rows = []
for line in open(r"bench\expert-tiering\anom-cpy3.err", encoding="utf-8", errors="ignore"):
    m = re.match(r"cpy: split=(\d+) src=(\S+) dst=(\S+) bytes=(\d+) us=(\d+) sync_us=(\d+) .*name=(\S+)", line)
    if m:
        split, srcbk, dstbk, by, us, sync, name = m.groups()
        by, us, sync = int(by), int(us), int(sync)
        if dstbk != "CPU" and by > 1000000:  # big host->device weight copies
            rows.append((split, name, by/1048576.0, us/1000.0, sync/1000.0))
            tot_bytes += by; tot_h2d_us += us; n += 1
if rows:
    print(f"big h2d copies: {n}, bytes {tot_bytes/1048576.0:.0f} MiB, host time {tot_h2d_us/1000.0:.0f} ms -> {tot_bytes/tot_h2d_us/1048.576:.2f} GB/s")
    for r in rows[:6]: print("  split %s %s %.1f MiB host %.1f ms sync %.1f ms" % r)
    # distribution of sync
    syncs = sorted(r[4] for r in rows)
    print("sync ms: min %.1f med %.1f max %.1f" % (syncs[0], syncs[len(syncs)//2], syncs[-1]))
