#!/usr/bin/env python3
"""Summarize MTP acceptance from llama-cli logs (M1-3b).

Reads the stderr logs of a draft-mtp run and prints the counters needed for the
M1 gate. The authoritative numbers come from the last server timings JSON; the
per-cycle "add accepted tokens" lines are used as a cross-check because they can
miss the last bonus token when generation stops on EOG.

    python bench/tools/mtp_accept.py bench/raw/m1-3b/res-k4-r1.err [more.err ...]

Output columns: file, predicted_n, draft_n, accepted_n, cycles, E, p, and the
cross-check cycles/E from the per-cycle lines.
"""
import json
import re
import sys
from pathlib import Path

TIMINGS = re.compile(r'"timings":(\{.*?\})\}')
CYCLE = re.compile(r"add accepted tokens: sampled=\d+, ids\.size=(\d+), n_draft=(\d+)")


def last_timings(text):
    best = None
    for m in TIMINGS.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if "draft_n" in obj:
            best = obj
    return best


def main(paths):
    print("file,predicted_n,draft_n,accepted_n,cycles,E,p,cycle_lines,cycle_draft,cycle_accepted,cycle_E")
    for p in paths:
        path = Path(p)
        text = path.read_text(encoding="utf-8", errors="replace")
        t = last_timings(text)
        if not t:
            print("%s,,,,,,,,,,," % path.name)
            continue
        pred = t["predicted_n"]
        draft = t.get("draft_n", 0)
        acc = t.get("draft_n_accepted", 0)
        cycles = pred - acc
        e = pred / cycles if cycles else 0.0
        prob = acc / draft if draft else 0.0
        lines = CYCLE.findall(text)
        c_draft = sum(int(d) for _, d in lines)
        c_acc = sum(int(n) - 1 for n, _ in lines)
        c_cycles = len(lines)
        c_e = (c_acc + c_cycles) / c_cycles if c_cycles else 0.0
        print("%s,%d,%d,%d,%d,%.3f,%.3f,%d,%d,%d,%.3f" % (
            path.name, pred, draft, acc, cycles, e, prob,
            c_cycles, c_draft, c_acc, c_e))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
