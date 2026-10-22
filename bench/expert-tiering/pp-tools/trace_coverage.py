"""Expert-coverage analyzer for the Qwen3.6 35B A3B prefill study.

Two input formats:

1) LLAMA_MOE_TRACE binary (decode / ubatch <= 32 only):
   record = int32 layer, int32 n, int32 ids[n]; ids = flat [n_expert_used, n_tokens],
   one record per MoE-cache lookup, so the layer id wraps at each token.
   The first record set is the first token (earliest -> latest layer in graph order).

2) GGML_MOE_COPY_TRACE text (the prefill path, any ubatch):
   one line per host expert tensor per eval:
     moe_copy: blk.7.ffn_gate_exps.weight host=1.79 MiB bank=0.00 MiB
   Distinct experts of a layer = (host + bank) / per-expert-layer bytes, because
   sched_copy_experts uploads or bank-copies exactly the selected experts (plus a
   <= 512 B padding on the last expert of a layer). Whole-layer-resident layers
   never reach this callback: no line, coverage 1.0 by construction.

Usage:
  python trace_coverage.py --bin <trace.bin> [--n-expert 256] [--top-k 8]
  python trace_coverage.py --moe-copy <run.log.err|run.log> [--n-expert 256]
"""
import argparse
import array
import os
import re
import sys
from collections import Counter, defaultdict

PER_EXPERT_LAYER_MIB = 1.820  # gate + up + down, from gguf_dims_result.txt


def read_binary(path):
    with open(path, "rb") as f:
        data = f.read()
    a = array.array("i")
    a.frombytes(data[: len(data) // a.itemsize * a.itemsize])
    recs = []
    i = 0
    while i + 2 <= len(a):
        il, n = a[i], a[i + 1]
        i += 2
        if n < 0 or i + n > len(a):
            print("truncated record at %d" % i)
            break
        recs.append((il, list(a[i:i + n])))
        i += n
    return recs


def split_tokens(recs):
    """A token ends when the layer id does not increase."""
    tokens, cur, last = [], [], None
    for il, ids in recs:
        if last is not None and il <= last:
            tokens.append(cur)
            cur = []
        cur.append((il, ids))
        last = il
    if cur:
        tokens.append(cur)
    return tokens


def report_binary(path, n_expert, top_k):
    recs = read_binary(path)
    print("file: %s" % path)
    print("records: %d" % len(recs))
    if not recs:
        print("EMPTY: no records (prefill bypasses the traced path; see README)")
        return
    tokens = [t for t in split_tokens(recs) if t]
    print("tokens: %d" % len(tokens))

    first = tokens[0]
    print("\nfirst record set (one token), %d layers:" % len(first))
    print("  layer  tokens  distinct  coverage")
    tot_ids = tot_distinct = 0
    for il, ids in first:
        n_tok = len(ids) // top_k if top_k else 0
        d = len(set(ids))
        tot_ids += len(ids)
        tot_distinct += d
        print("  %5d  %6d  %8d  %8.3f" % (il, n_tok, d, d / n_expert))
    print("  aggregate: ids=%d distinct=%d coverage=%.3f" %
          (tot_ids, tot_distinct, tot_distinct / max(1, len(first) * n_expert)))

    cov = []
    for tok in tokens:
        for il, ids in tok:
            cov.append(len(set(ids)) / n_expert)
    print("\nper-layer distinct over all %d tokens:" % len(tokens))
    per_layer = defaultdict(list)
    for tok in tokens:
        for il, ids in tok:
            per_layer[il].append(len(set(ids)) / n_expert)
    print("  layer  min    mean   max")
    for il in sorted(per_layer):
        v = per_layer[il]
        print("  %5d  %5.3f  %5.3f  %5.3f" % (il, min(v), sum(v) / len(v), max(v)))
    if cov:
        print("  overall coverage: min=%.3f mean=%.3f max=%.3f" %
              (min(cov), sum(cov) / len(cov), max(cov)))


RE_MOE = re.compile(r"moe_copy: (\S+) host=([0-9.]+) MiB bank=([0-9.]+) MiB")
RE_BLK = re.compile(r"blk\.(\d+)\.")

# Qwen3.6 35B: expert_ff = 512, n_embd = 2048, so one expert of one tensor is
#   gate/up: 2048 values/row, Q4_K = 8 x 144 B = 1152 B/row, 512 rows = 589824 B
#   down   : 2048 values/row, Q6_K = 8 x 210 B = 1680 B/row, 512 rows = 860160 B
BYTES_GATE_UP = 589824
BYTES_DOWN_Q6K = 860160
MIB = 1048576.0


def report_moe_copy(path, n_expert):
    # lines come in graph order, so the layer id wraps at each eval
    evals = []      # list of dicts: layer -> {tensor: (host, bank)} in MiB
    cur = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    last = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_MOE.search(line)
            if not m:
                continue
            name, host, bank = m.group(1), float(m.group(2)), float(m.group(3))
            mb = RE_BLK.search(name)
            if not mb:
                continue
            il = int(mb.group(1))
            # one layer has several tensors (same il), so an eval ends only when the id decreases
            if last is not None and il < last and cur:
                evals.append(dict(cur))
                cur = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
            tensor = name.split(".", 2)[-1].replace(".weight", "")
            cur[il][tensor][0] += host
            cur[il][tensor][1] += bank
            last = il
    if cur:
        evals.append(dict(cur))

    if not evals:
        print("no moe_copy lines in %s" % path)
        return
    print("file: %s" % path)
    print("eval record sets: %d" % len(evals))
    for i, ev in enumerate(evals, 1):
        host_mib = sum(v[0] for lay in ev.values() for v in lay.values())
        bank_mib = sum(v[1] for lay in ev.values() for v in lay.values())
        print("\n record set %d: traced layers = %d, host = %.1f MiB, bank = %.1f MiB, total = %.1f MiB"
              % (i, len(ev), host_mib, bank_mib, host_mib + bank_mib))
        print("  layer  distinct  coverage  down_type  gather MiB  whole-layer MiB  ratio")
        covs, gather_tot, whole_tot = [], 0.0, 0.0
        for il in sorted(ev):
            counts = {}
            # gate/up are Q4_K without exception, so they fix the expert count of the layer
            ref = None
            for tensor, (h, b) in ev[il].items():
                if tensor.startswith("ffn_gate") or tensor.startswith("ffn_up"):
                    ref = round((h + b) * MIB / BYTES_GATE_UP)
            for tensor, (h, b) in ev[il].items():
                tot = (h + b) * MIB
                if tensor.startswith("ffn_down"):
                    # Q6_K down: 860160 B/expert, Q4_K down: 589824 B/expert
                    sz = BYTES_GATE_UP
                    if ref:
                        d = tot / ref
                        sz = BYTES_DOWN_Q6K if abs(d - BYTES_DOWN_Q6K) < abs(d - BYTES_GATE_UP) else BYTES_GATE_UP
                    dtype = "Q6_K" if sz == BYTES_DOWN_Q6K else "Q4_K"
                else:
                    sz, dtype = BYTES_GATE_UP, None
                counts[tensor] = (round(tot / sz), sz, dtype)
            distinct = ref if ref else max(c[0] for c in counts.values())
            whole = sum(n_expert * c[1] for c in counts.values()) / MIB
            gather = sum(c[0] * c[1] for c in counts.values()) / MIB
            gather_tot += gather
            whole_tot += whole
            covs.append(min(1.0, distinct / n_expert))
            down_type = next((c[2] for c in counts.values() if c[2]), "?")
            print("  %5d  %8d  %8.3f  %9s  %10.1f  %15.1f  %5.2f"
                  % (il, distinct, distinct / n_expert, down_type, gather, whole, gather / whole))
        n_host_layers = len(ev)
        print("  traced-layer coverage: min=%.3f mean=%.3f max=%.3f"
              % (min(covs), sum(covs) / len(covs), max(covs)))
        print("  traced layers: gather = %.1f MiB, whole-layer = %.1f MiB, whole/gather = %.2fx"
              % (gather_tot, whole_tot, whole_tot / gather_tot))
        print("  NOTE: layers with no moe_copy line are whole-layer bank resident (no upload,"
              " coverage 1.0) or MTP/unused blocks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin")
    ap.add_argument("--moe-copy")
    ap.add_argument("--n-expert", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    a = ap.parse_args()
    if a.bin:
        report_binary(a.bin, a.n_expert, a.top_k)
    if a.moe_copy:
        report_moe_copy(a.moe_copy, a.n_expert)
    if not a.bin and not a.moe_copy:
        ap.error("give --bin or --moe-copy")


if __name__ == "__main__":
    main()
