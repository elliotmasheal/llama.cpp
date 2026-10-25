"""Metrics for two logits dumps: PPL, mean KLD, same top-1, first divergence.

The KLD definition follows tools/perplexity/perplexity.cpp: base probabilities
come from row A, the sum runs over the vocabulary entries A keeps (log p > -16,
i.e. 16 nats below the row max). One difference: the sum is divided by the kept
mass, so the number is the KLD of a proper distribution and cannot go slightly
negative on quantisation noise. The dropped tail is reported, because it is the
only place where the side with a wider tail can hide mass.
"""

import numpy as np

from logitio import DumpError


def _pairs(a, b, align):
    """Yield matched (Pos_a, Pos_b) in row order, optionally intersecting."""
    if not align:
        if a.n_pos != b.n_pos:
            raise DumpError("row counts differ: %d vs %d (use --align to intersect)"
                            % (a.n_pos, b.n_pos))
        for pa, pb in zip(a.positions(), b.positions()):
            if pa.offset != pb.offset:
                raise DumpError("row offsets diverge at %s vs %s (use --align)"
                                % (pa.offset, pb.offset))
            yield pa, pb
        return

    # both streams are sorted by offset, so a two-cursor walk yields the
    # intersection in row order
    oa = a.offsets()
    ob = b.offsets()
    ga = ((int(o), p) for o, p in zip(oa.tolist(), a.positions()))
    gb = ((int(o), p) for o, p in zip(ob.tolist(), b.positions()))
    na = next(ga, None)
    nb = next(gb, None)
    while na is not None and nb is not None:
        if na[0] == nb[0]:
            yield na[1], nb[1]
            na = next(ga, None)
            nb = next(gb, None)
        elif na[0] < nb[0]:
            na = next(ga, None)
        else:
            nb = next(gb, None)


def ppl_of(d, nats_floor=16.0):
    """PPL of one dump over its rows, conditioned on the kept vocabulary."""
    n = 0
    nll = 0.0
    mass = 0.0
    clamp = 0.0
    for pos in d.positions():
        lp = pos.logp
        keep = lp > -nats_floor
        m = float(np.sum(np.exp(lp[keep].astype(np.float64))))
        if m > 0:
            nll += -(float(lp[pos.target]) - float(np.log(m)))
        mass += m
        clamp += float(np.sum(np.exp(lp[~keep].astype(np.float64))))
        n += 1
    if n == 0:
        raise DumpError("no rows")
    return {"rows": n, "ppl": float(np.exp(nll / n)), "mean_nll": nll / n,
            "kept_mass": mass / n, "clamp_mass": clamp / n}


def compare(a, b, kld_max=1e-4, margin_max=0.05, nats_floor=16.0, align=False):
    """Compare dump A (base) against dump B (test). Returns a report dict."""
    n = 0
    nll_a = 0.0
    nll_b = 0.0
    kld_sum = 0.0
    kld_rows = []
    mass_cover = 0.0
    clamp_mass = 0.0
    same_top = 0
    dp2 = 0.0
    diverg = []
    bad_target = 0
    pa_first = pb_first = None

    for pa, pb in _pairs(a, b, align):
        if pa_first is None:
            pa_first, pb_first = pa.offset, pb.offset
        la = pa.logp
        lb = pb.logp
        tv = pa.target
        if pb.target != tv:
            bad_target += 1

        # condition both rows on the kept vocabulary: PPL and KLD then share a
        # denominator and the fork writer's clamped-at-the-floor mass cancels
        keep = la > -nats_floor
        ka = la[keep].astype(np.float64)
        kb = lb[keep].astype(np.float64)
        wa = np.exp(ka)
        wb = np.exp(kb)
        ma = float(wa.sum())
        mb = float(wb.sum())
        mass_cover += ma
        clamp_mass += float(np.sum(np.exp(la[~keep].astype(np.float64))))
        if ma > 0 and mb > 0:
            log_ma = np.log(ma)
            log_mb = np.log(mb)
            ka = ka - log_ma
            kb = kb - log_mb
            d = float(np.sum(wa * (ka - kb)) / ma)
        else:
            log_ma = log_mb = 0.0
            d = 0.0
        nll_a += -(float(la[tv]) - log_ma)
        nll_b += -(float(lb[tv]) - log_mb)
        kld_sum += d
        kld_rows.append(d)

        ta = int(np.argmax(la))
        tb = int(np.argmax(lb))
        if ta == tb:
            same_top += 1
        else:
            diverg.append((pa.offset, ta, tb, float(la[ta] - la[tb])))

        dp = float(np.exp(float(lb[tv]) - log_mb) - np.exp(float(la[tv]) - log_ma))
        dp2 += dp * dp
        n += 1

    if n == 0:
        raise DumpError("no comparable rows")

    rows = np.asarray(kld_rows, dtype=np.float64)
    mean_kld = kld_sum / n
    first_div = diverg[0] if diverg else None
    worst = max(diverg, key=lambda d: d[3]) if diverg else None
    flips_above = [d for d in diverg if d[3] >= margin_max]

    report = {
        "rows_compared": n,
        "first_offset": pa_first,
        "tokens_match": bad_target == 0,
        "bad_targets": bad_target,
        "ppl_a": float(np.exp(nll_a / n)),
        "ppl_b": float(np.exp(nll_b / n)),
        "ln_ppl_ratio": float((nll_b - nll_a) / n),
        "mean_kld": float(mean_kld),
        "kld_p95": float(np.percentile(rows, 95)),
        "kld_row_max": float(rows.max()),
        "kept_mass": float(mass_cover / n),
        "clamp_mass": float(clamp_mass / n),
        "same_top_pct": 100.0 * same_top / n,
        "n_diverg": len(diverg),
        "first_div": None if first_div is None else {
            "offset": first_div[0], "base_top": first_div[1],
            "test_top": first_div[2], "margin": first_div[3],
        },
        "worst_div_margin": None if worst is None else worst[3],
        "dp_rms_pct": 100.0 * float(np.sqrt(dp2 / n)),
        "kld_max": kld_max,
        "margin_max": margin_max,
    }
    reasons = []
    if not report["tokens_match"]:
        reasons.append("%d rows predict different tokens (dumps are not from the same token stream)"
                       % bad_target)
    if mean_kld > kld_max:
        reasons.append("mean KLD %.3e > %.3e" % (mean_kld, kld_max))
    if flips_above:
        reasons.append("%d top-1 flips at margin >= %.3f nats (worst %.4f)"
                       % (len(flips_above), margin_max, max(d[3] for d in flips_above)))
    report["fail_reasons"] = reasons
    report["verdict"] = "PASS" if not reasons else "FAIL"
    return report


def verify_dump(path, nats_floor=16.0):
    """Structural self-check of one dump: header, size, row normalisation.

    A wrong nv/offset/layout guess shows up as a row mass far from 1, so this
    is the cheapest way to tell "the parser understands this file" from "the
    numbers are plausible".
    """
    from logitio import open_dump

    with open_dump(path) as d:
        info = d.describe()
        rows = 0
        mass_lo = mass_hi = None
        top_lo = top_hi = None
        clamp_hi = 0.0
        for pos in d.positions():
            lp = pos.logp
            keep = lp > -nats_floor
            m = float(np.sum(np.exp(lp.astype(np.float64))))
            mass_lo = m if mass_lo is None else min(mass_lo, m)
            mass_hi = m if mass_hi is None else max(mass_hi, m)
            clamp_hi = max(clamp_hi, float(np.sum(np.exp(lp[~keep].astype(np.float64)))))
            t = float(lp.max())
            top_lo = t if top_lo is None else min(top_lo, t)
            top_hi = t if top_hi is None else max(top_hi, t)
            rows += 1
    # the fork floors every entry 16 nats below the row max, so a peaked row
    # invents up to n_vocab * exp(-16) of mass at that floor: allow for it
    slack = 1.05 * d.n_vocab * np.exp(-nats_floor)
    reasons = []
    if rows != d.n_pos:
        reasons.append("decoded %d rows, header says %d" % (rows, d.n_pos))
    if mass_hi is None:
        reasons.append("no rows")
    else:
        if mass_hi > 1.0 + slack:
            reasons.append("row mass %.6f > %.6f (wrong offset or vocabulary width)"
                           % (mass_hi, 1.0 + slack))
        if mass_lo < 0.90:
            reasons.append("row mass %.6f < 0.90 (too much mass below the floor or a bad scale)" % mass_lo)
        if top_hi > 1e-3 or top_lo < -30.0:
            reasons.append("row max log-prob in [%.3f, %.3f], expected (-30, 0]" % (top_lo, top_hi))
    info.update({"rows_checked": rows, "row_mass_min": mass_lo, "row_mass_max": mass_hi,
                 "row_mass_limit": 1.0 + slack, "clamp_mass_max": clamp_hi,
                 "row_top_logp_min": top_lo, "row_top_logp_max": top_hi,
                 "fail_reasons": reasons,
                 "verdict": "PASS" if not reasons else "FAIL"})
    return info


def format_report(ra, rb, rep):
    """Fixed-column text block: one screen, PPL | KLD | same top | verdict."""
    out = []
    out.append("dump A   : %s" % ra["path"])
    out.append("           fmt %s  n_vocab %d  rows %d" % (ra["fmt"], ra["n_vocab"], ra["rows"]))
    out.append("dump B   : %s" % rb["path"])
    out.append("           fmt %s  n_vocab %d  rows %d" % (rb["fmt"], rb["n_vocab"], rb["rows"]))
    out.append("rows     : %d compared, first offset %d, tokens %s"
               % (rep["rows_compared"], rep["first_offset"],
                  "identical" if rep["tokens_match"] else "MISMATCH"))
    out.append("PPL      : %.6f (A)  %.6f (B)   ln ratio %+.6f"
               % (rep["ppl_a"], rep["ppl_b"], rep["ln_ppl_ratio"]))
    out.append("mean KLD : %.4e  (p95 %.4e, max row %.4e, limit %.1e)"
               % (rep["mean_kld"], rep["kld_p95"], rep["kld_row_max"], rep["kld_max"]))
    out.append("kept mass: %.6f   clamped-to-floor mass %.3e (both sides renormalised)"
               % (rep["kept_mass"], rep["clamp_mass"]))
    out.append("same top : %.3f %%   (%d rows differ, worst margin %s)"
               % (rep["same_top_pct"], rep["n_diverg"],
                  "n/a" if rep["worst_div_margin"] is None else "%.4f nats" % rep["worst_div_margin"]))
    if rep["first_div"] is None:
        out.append("diverg   : none")
    else:
        fd = rep["first_div"]
        out.append("diverg   : first at offset %d, top-1 %d -> %d, margin %.4f nats"
                   % (fd["offset"], fd["base_top"], fd["test_top"], fd["margin"]))
    out.append("dP rms   : %.3f %% on the target token" % rep["dp_rms_pct"])
    for r in rep["fail_reasons"]:
        out.append("fail     : %s" % r)
    out.append("verdict  : %s" % rep["verdict"])
    return "\n".join(out)
