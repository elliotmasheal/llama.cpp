"""Quality gate for qwen36rt: PPL, mean KLD, same top-1, PASS/FAIL.

usage:
  python bench/tools/quality/qgate.py ppl     DUMP
  python bench/tools/quality/qgate.py compare A B [--align] [--kld-max X] [--margin-max X]
                                              [--json OUT] [--quiet]
  python bench/tools/quality/qgate.py verify  DUMP
  python bench/tools/quality/qgate.py convert DUMP --to-rt OUT [--rows N]
  python bench/tools/quality/qgate.py selftest

Exit codes: 0 PASS, 3 FAIL, 2 bad input or malformed dump.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logitio
import metrics


def cmd_ppl(args):
    try:
        with logitio.open_dump(args.dump) as d:
            info = d.describe()
            info.update(metrics.ppl_of(d))
    except logitio.DumpError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    print(json.dumps(info, indent=2))
    return 0


def cmd_compare(args):
    ra = rb = None
    try:
        with logitio.open_dump(args.a) as a, logitio.open_dump(args.b) as b:
            ra, rb = a.describe(), b.describe()
            if a.n_vocab != b.n_vocab:
                raise logitio.DumpError("vocabulary size differs: %d vs %d"
                                        % (a.n_vocab, b.n_vocab))
            rep = metrics.compare(a, b, kld_max=args.kld_max, margin_max=args.margin_max,
                                  align=args.align)
            rep.update({"dump_a": ra, "dump_b": rb})
    except logitio.DumpError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
            f.write("\n")
    if not args.quiet:
        print(metrics.format_report(ra, rb, rep))
    else:
        print("PPL %.6f | mean KLD %.4e | same top %.3f %% | %s"
              % (rep["ppl_a"], rep["mean_kld"], rep["same_top_pct"], rep["verdict"]))
    return 0 if rep["verdict"] == "PASS" else 3


def cmd_verify(args):
    """Check a dump against its own header: size, row normalisation, verdict."""
    try:
        info = metrics.verify_dump(args.dump)
    except logitio.DumpError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    for r in info.get("fail_reasons", []):
        print("fail     : %s" % r, file=sys.stderr)
    print(json.dumps(info, indent=2))
    return 0 if info["verdict"] == "PASS" else 3


def cmd_convert(args):
    with logitio.open_dump(args.dump) as d:
        if d.fmt != "fork":
            print("ERROR: convert expects a fork dump", file=sys.stderr)
            return 2
        offs_all = d.offsets()
        if args.chunk is None:
            if d.n_chunk > 1:
                print("ERROR: dump has %d chunks; pick one with --chunk N (0..%d)"
                      % (d.n_chunk, d.n_chunk - 1), file=sys.stderr)
                return 2
            chunk = 0
        else:
            chunk = args.chunk
            if not 0 <= chunk < d.n_chunk:
                print("ERROR: --chunk %d outside 0..%d" % (chunk, d.n_chunk - 1), file=sys.stderr)
                return 2
        rows = []
        offs = []
        for pos in d.positions():
            if pos.chunk != chunk:
                continue
            if args.rows and len(rows) >= args.rows:
                break
            offs.append(pos.offset)
            rows.append(pos.logp.astype(np.float32))
        if not rows:
            print("ERROR: chunk %d has no rows" % chunk, file=sys.stderr)
            return 2
        start = int(offs[0])
        tokens = d.tokens[start:start + len(rows) + 1]
        if len(tokens) != len(rows) + 1:
            raise logitio.DumpError("token range for the converted rows is outside the dump")
        logitio.write_runtime_dump(args.to_rt, tokens, np.stack(rows), start=start)
    print("wrote %s (%d rows, chunk %d, start %d)" % (args.to_rt, len(rows), chunk, start))
    return 0


def cmd_selftest(args):
    import selftest
    return selftest.main([])


def main(argv):
    p = argparse.ArgumentParser(prog="qgate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("ppl", help="PPL of one dump")
    q.add_argument("dump")
    q.set_defaults(fn=cmd_ppl)

    q = sub.add_parser("compare", help="compare two dumps (A is the base)")
    q.add_argument("a")
    q.add_argument("b")
    q.add_argument("--align", action="store_true",
                   help="compare the intersection of row offsets instead of demanding equality")
    q.add_argument("--kld-max", type=float, default=1e-4, help="KLD limit (default 1e-4)")
    q.add_argument("--margin-max", type=float, default=0.05,
                   help="allowed greedy margin at a top-1 flip (default 0.05 nats, U-7)")
    q.add_argument("--json", help="write the full report here")
    q.add_argument("--quiet", action="store_true", help="one line only")
    q.set_defaults(fn=cmd_compare)

    q = sub.add_parser("verify", help="header, row mass and re-quantisation checks")
    q.add_argument("dump")
    q.set_defaults(fn=cmd_verify)

    q = sub.add_parser("convert", help="fork dump -> rt dump (compatibility path)")
    q.add_argument("dump")
    q.add_argument("--to-rt", required=True)
    q.add_argument("--chunk", type=int, default=None, help="chunk to convert (default: the only one)")
    q.add_argument("--rows", type=int, default=0, help="keep at most this many rows of that chunk")
    q.set_defaults(fn=cmd_convert)

    q = sub.add_parser("selftest", help="synthetic format and metric tests")
    q.set_defaults(fn=cmd_selftest)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except logitio.DumpError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
