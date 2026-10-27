"""Summarize GGML_VK_PERF_LOGGER tables: per-graph total and top ops.

usage: python bench/tools/summarize_perf.py <stderr-log> [top]
       python bench/tools/summarize_perf.py <stderr-log> --agg [top]

--agg sums every key across all tables, once by base op name and once by the exact
shape key. Absolute us are inflated ~2.6x by the logger, so read shares, not ms.
"""
import re
import sys
from collections import OrderedDict

path = sys.argv[1]
agg = len(sys.argv) > 2 and sys.argv[2] == "--agg"
top = int(sys.argv[3]) if agg and len(sys.argv) > 3 else (int(sys.argv[2]) if len(sys.argv) > 2 else 6)

tables = []
cur = None
for line in open(path, errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("Vulkan Timings:"):
        cur = []
        continue
    if cur is None:
        continue
    m = re.match(r"^Total time: ([\d.]+) us", line)
    if m:
        tables.append((float(m.group(1)), cur))
        cur = None
        continue
    m = re.match(r"^(.*?): (\d+) x ([\d.]+) us = ([\d.]+) us", line)
    if m:
        cur.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), line))

print("tables: %d" % len(tables))

if agg:
    by_base = OrderedDict()
    by_key = OrderedDict()
    for total, rows in tables:
        for name, n, per, tot, _ in rows:
            # keys carry a shape/type suffix ("MUL_MAT_VEC q4_K m=..", "RMS_NORM(128,16,2,1)")
            base = re.match(r"^[A-Z0-9_]+", name)
            base = base.group(0) if base else name
            for d, k in ((by_base, base), (by_key, name)):
                e = d.setdefault(k, [0, 0.0])
                e[0] += n
                e[1] += tot
    grand = sum(v[1] for v in by_base.values())
    print("grand total: %.1f ms (logger inflated, shares only)" % (grand / 1000.0))
    print("-- by op --")
    for k, (n, t) in sorted(by_base.items(), key=lambda r: -r[1][1]):
        print("%7.1f ms %5.1f%% %8d x  %s" % (t / 1000.0, 100.0 * t / grand, n, k))
    print("-- top %d exact keys --" % top)
    for k, (n, t) in sorted(by_key.items(), key=lambda r: -r[1][1])[:top]:
        print("%7.1f ms %5.1f%% %8d x  %s" % (t / 1000.0, 100.0 * t / grand, n, k[:90]))
    sys.exit(0)

for i, (total, rows) in enumerate(tables):
    rows = sorted(rows, key=lambda r: -r[3])
    head = ", ".join("%s %.2fms" % (r[0][:38], r[3] / 1000.0) for r in rows[:top])
    print("[%02d] total %8.2f ms | %s" % (i, total / 1000.0, head))

