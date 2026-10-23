"""Summarize GGML_VK_PERF_LOGGER tables: per-graph total and top ops.

usage: python bench/tools/summarize_perf.py <stderr-log> [top]
"""
import re
import sys
from collections import OrderedDict

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 6

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
for i, (total, rows) in enumerate(tables):
    rows = sorted(rows, key=lambda r: -r[3])
    head = ", ".join("%s %.2fms" % (r[0][:38], r[3] / 1000.0) for r in rows[:top])
    print("[%02d] total %8.2f ms | %s" % (i, total / 1000.0, head))
