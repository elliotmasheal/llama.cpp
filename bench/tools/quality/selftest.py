"""Synthetic tests for the dump readers and the KLD/PPL metrics.

Every case builds the fixture from known float32 logits, so the expected
numbers are computed independently in float64 and the tool has to match them.
Run: python bench/tools/quality/selftest.py   (exit 0 on success)
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logitio
import metrics
from logitio import DumpError

N_VOCAB = 512
N_CTX = 32
N_CHUNK = 3
FIRST = N_CTX // 2
REC = N_CTX - 1 - FIRST


def ref_log_softmax(rows):
    m = rows.max(axis=1, keepdims=True)
    s = np.log(np.sum(np.exp(rows - m), axis=1, keepdims=True))
    return rows - m - s


def make_rows(rng, n_rows, n_vocab=N_VOCAB, scale=4.0):
    """Logits with a wide tail, so the 16-nat floor actually cuts something."""
    return (rng.normal(size=(n_rows, n_vocab)) * scale).astype(np.float32)


def make_tokens(rng, n):
    return rng.integers(0, N_VOCAB, size=n).astype(np.int32)


def point_tokens_at_rows(rows, tokens):
    """Make each predicted token a plausible sample (the row's argmax).

    Real text predicts near-top tokens; a random token would sit below the
    16-nat floor and PPL would be meaningless.
    """
    tokens = tokens.copy()
    offsets = np.array([c * N_CTX + FIRST + j for c in range(N_CHUNK) for j in range(REC)])
    for i, o in enumerate(offsets):
        tokens[o + 1] = int(np.argmax(rows[i]))
    return tokens


def fork_fixture(tmp, rows, tokens):
    p = os.path.join(tmp, "fork.bin")
    logitio.write_fork_dump(p, N_CTX, tokens, rows)
    return p


def test_fork_roundtrip(tmp, rng):
    rows = make_rows(rng, REC * N_CHUNK)
    tokens = make_tokens(rng, N_CTX * N_CHUNK)
    path = fork_fixture(tmp, rows, tokens)
    ref = ref_log_softmax(rows.astype(np.float64))
    offsets = np.array([c * N_CTX + FIRST + j for c in range(N_CHUNK) for j in range(REC)])
    want_targets = tokens[offsets + 1]
    with logitio.ForkDump(path) as d:
        assert d.n_pos == REC * N_CHUNK, d.n_pos
        assert np.array_equal(d.offsets(), offsets)
        got = []
        for pos in d.positions():
            got.append(pos.target)
        assert np.array_equal(np.array(got), want_targets), "targets do not follow the prompt"
    # entries above the 16-nat floor are quantised; the rest are clamped to it
    with logitio.ForkDump(path) as d:
        for i, pos in enumerate(d.positions()):
            row = rows[i].astype(np.float64)
            min_l = max(float(row.min()), float(row.max()) - 16.0)
            span = float(row.max()) - min_l
            half = span / 65535.0 / 2.0 + 5e-6
            inside = row > min_l
            e = float(np.abs(pos.logp.astype(np.float64)[inside] - ref[i][inside]).max())
            assert e <= half, "row %d error %.3e > half step %.3e" % (i, e, half)
            floor = float(pos.logp[~inside].min()) if not inside.all() else 0.0
            if not inside.all():
                assert floor <= -16.0, "clamped value %.3f is above the floor" % floor
    return "fork roundtrip: %d rows, dequant error <= half step" % (REC * N_CHUNK)


def test_fork_verify(tmp, rng):
    rows = make_rows(rng, REC * N_CHUNK)
    tokens = make_tokens(rng, N_CTX * N_CHUNK)
    path = fork_fixture(tmp, rows, tokens)
    info = metrics.verify_dump(path)
    assert info["verdict"] == "PASS", info["fail_reasons"]
    assert 0.99 < info["row_mass_min"] <= info["row_mass_max"] <= info["row_mass_limit"], info["row_mass_max"]

    # a zeroed scale word makes the row decode to a constant: mass collapses
    with open(path, "r+b") as f:
        f.seek(20 + 4 * N_CTX * N_CHUNK)
        f.write(np.float32(0.0).tobytes())
    info = metrics.verify_dump(path)
    assert info["verdict"] == "FAIL", "a zeroed scale word was accepted"

    # a header that disagrees with the body is refused, not silently truncated
    with open(path, "r+b") as f:
        f.seek(12)
        f.write(np.uint32(N_VOCAB + 2).tobytes())
    try:
        metrics.verify_dump(path)
    except DumpError:
        pass
    else:
        raise AssertionError("n_vocab tampered with was accepted")
    return "verify: clean dump passes, zeroed scale and bad n_vocab rejected"


def test_rt_roundtrip(tmp, rng):
    rows = make_rows(rng, 16)
    tokens = make_tokens(rng, 17)
    path = os.path.join(tmp, "rt.bin")
    logitio.write_runtime_dump(path, tokens, rows, start=5)
    ref = ref_log_softmax(rows.astype(np.float64))
    with logitio.RuntimeDump(path) as d:
        assert d.n_pos == 16 and d.start == 5
        assert np.array_equal(d.offsets(), np.arange(5, 21))
        assert d.tokens.size == 17
        for j, pos in enumerate(d.positions()):
            assert pos.target == int(tokens[j + 1])
            assert float(np.abs(pos.logp - ref[j]).max()) < 2e-6, j
    return "rt roundtrip: 16 rows, log-softmax within 2e-6"


def test_identical(tmp, rng):
    rows = make_rows(rng, REC * N_CHUNK)
    tokens = make_tokens(rng, N_CTX * N_CHUNK)
    path = fork_fixture(tmp, rows, tokens)
    with logitio.ForkDump(path) as a, logitio.ForkDump(path) as b:
        rep = metrics.compare(a, b)
    assert rep["mean_kld"] == 0.0, rep["mean_kld"]
    assert rep["same_top_pct"] == 100.0
    assert rep["verdict"] == "PASS", rep["fail_reasons"]
    return "identical dumps: KLD 0, same top 100 %, PASS"


def test_kld_against_reference(tmp, rng):
    base = make_rows(rng, REC * N_CHUNK)
    tokens = point_tokens_at_rows(base, make_tokens(rng, N_CTX * N_CHUNK))
    pa = fork_fixture(tmp, base, tokens)
    noise = (rng.normal(size=base.shape) * 0.35).astype(np.float32)
    test = base + noise
    pb = os.path.join(tmp, "fork-test.bin")
    logitio.write_fork_dump(pb, N_CTX, tokens, test)

    # independent reference in float64 from the untruncated logits
    lp_a = ref_log_softmax(base.astype(np.float64))
    lp_b = ref_log_softmax(test.astype(np.float64))
    p_a = np.exp(lp_a)
    keep = lp_a > -16.0
    norm_a = np.sum(np.where(keep, p_a, 0.0), axis=1, keepdims=True)
    lm_a = np.log(norm_a)
    p_b = np.exp(lp_b)
    norm_b = np.sum(np.where(keep, p_b, 0.0), axis=1, keepdims=True)
    lm_b = np.log(norm_b)
    ref_kld = float(np.mean(np.sum(np.where(keep, p_a * (lp_a - lp_b), 0.0), axis=1) / norm_a[:, 0]))
    offsets = np.array([c * N_CTX + FIRST + j for c in range(N_CHUNK) for j in range(REC)])
    tgt = tokens[offsets + 1]
    ref_ppl = float(np.exp(np.mean(-(lp_a[np.arange(len(base)), tgt] - lm_a[:, 0]))))
    same = float(np.mean(np.argmax(lp_a, axis=1) == np.argmax(lp_b, axis=1)) * 100.0)

    with logitio.ForkDump(pa) as a, logitio.ForkDump(pb) as b:
        rep = metrics.compare(a, b)
    rel = abs(rep["mean_kld"] - ref_kld) / ref_kld
    assert rel < 0.02, "KLD %.6e vs reference %.6e (%.2f %%)" % (rep["mean_kld"], ref_kld, 100 * rel)
    assert abs(rep["ppl_a"] - ref_ppl) / ref_ppl < 1e-3, "PPL %.6f vs %.6f" % (rep["ppl_a"], ref_ppl)
    assert abs(rep["same_top_pct"] - same) < 1e-9, "same top %.3f vs %.3f" % (rep["same_top_pct"], same)
    assert rep["verdict"] == "FAIL", "a 0.35-logit perturbation must not pass"
    return ("KLD %.4e vs reference %.4e (%.2f %% off), PPL %.4f, same top %.1f %%, FAIL as expected"
            % (rep["mean_kld"], ref_kld, 100 * rel, rep["ppl_a"], rep["same_top_pct"]))


def test_small_perturbation_passes(tmp, rng):
    base = make_rows(rng, REC * N_CHUNK)
    tokens = make_tokens(rng, N_CTX * N_CHUNK)
    pa = fork_fixture(tmp, base, tokens)
    pb = os.path.join(tmp, "fork-tiny.bin")
    logitio.write_fork_dump(pb, N_CTX, tokens, base + (rng.normal(size=base.shape) * 1e-3).astype(np.float32))
    with logitio.ForkDump(pa) as a, logitio.ForkDump(pb) as b:
        rep = metrics.compare(a, b)
    assert rep["verdict"] == "PASS", rep["fail_reasons"]
    assert rep["mean_kld"] < 1e-4, rep["mean_kld"]
    assert 0.0 < rep["dp_rms_pct"] < 5.0, "target-probability rms %.3f %%" % rep["dp_rms_pct"]
    return "1e-3 logit noise: KLD %.3e, same top %.1f %%, dP rms %.3f %%, PASS" % (
        rep["mean_kld"], rep["same_top_pct"], rep["dp_rms_pct"])


def test_floor_and_tail(tmp, rng):
    """A wide row loses its tail at the 16-nat floor; the tool must report the
    dropped mass instead of silently shortening the row."""
    rows = make_rows(rng, REC * N_CHUNK, scale=6.0)
    rows[:, 0] += 30.0
    tokens = make_tokens(rng, N_CTX * N_CHUNK)
    path = fork_fixture(tmp, rows, tokens)
    with logitio.ForkDump(path) as a, logitio.ForkDump(path) as b:
        rep = metrics.compare(a, b)
        d = a.describe()
    assert d["n_vocab"] == N_VOCAB
    assert 0.0 <= rep["clamp_mass"] < 1.0, rep["clamp_mass"]
    assert rep["verdict"] == "PASS", rep["fail_reasons"]
    return "16-nat floor: clamped mass %.3e, kept mass %.6f, vocab width kept" % (
        rep["clamp_mass"], rep["kept_mass"])


def test_cross_format_align(tmp, rng):
    rows = make_rows(rng, REC * N_CHUNK)
    tokens = make_tokens(rng, N_CTX * N_CHUNK)
    pa = fork_fixture(tmp, rows, tokens)
    # a full-prompt rt dump over the same tokens: all positions, chunked view lost
    lp = ref_log_softmax(rows.astype(np.float32))
    n_tok = N_CTX * N_CHUNK
    full = np.zeros((n_tok - 1, N_VOCAB), dtype=np.float32)
    for c in range(N_CHUNK):
        for j in range(REC):
            full[c * N_CTX + FIRST + j] = lp[c * REC + j]
    pb = os.path.join(tmp, "rt-align.bin")
    logitio.write_runtime_dump(pb, tokens, full)
    with logitio.ForkDump(pa) as a, logitio.RuntimeDump(pb) as b:
        strict_failed = False
        try:
            metrics.compare(a, b)
        except DumpError:
            strict_failed = True
        rep = metrics.compare(a, b, align=True)
    assert strict_failed, "strict compare should reject the different row sets"
    assert rep["rows_compared"] == REC * N_CHUNK, rep["rows_compared"]
    assert rep["mean_kld"] < 2e-4, rep["mean_kld"]
    return ("align: %d rows of %d intersected, KLD %.3e (quantisation only)"
            % (rep["rows_compared"], N_CTX * N_CHUNK, rep["mean_kld"]))


def test_malformed(tmp, rng):
    rows = make_rows(rng, REC * N_CHUNK)
    tokens = make_tokens(rng, N_CTX * N_CHUNK)
    good = fork_fixture(tmp, rows, tokens)
    cases = []

    bad = os.path.join(tmp, "bad-magic.bin")
    data = open(good, "rb").read()
    open(bad, "wb").write(b"_logits!" + data[8:])
    cases.append(("wrong magic", bad, logitio.ForkDump))

    bad = os.path.join(tmp, "truncated.bin")
    open(bad, "wb").write(data[:len(data) - 100])
    cases.append(("truncated body", bad, logitio.ForkDump))

    bad = os.path.join(tmp, "short-hdr.bin")
    open(bad, "wb").write(data[:10])
    cases.append(("short header", bad, logitio.ForkDump))

    bad = os.path.join(tmp, "bad-vocab.bin")
    open(bad, "wb").write(data[:12] + np.int32(-5).tobytes() + data[16:])
    cases.append(("negative n_vocab", bad, logitio.ForkDump))

    for name, path, opener in cases:
        try:
            d = opener(path)
        except DumpError:
            continue
        d.close()
        raise AssertionError("%s was accepted" % name)

    # token stream mismatch between two dumps must fail the gate, not crash
    other = list(tokens)
    other[N_CTX * N_CHUNK - 1] = (other[N_CTX * N_CHUNK - 1] + 1) % N_VOCAB
    pb = os.path.join(tmp, "fork-other.bin")
    logitio.write_fork_dump(pb, N_CTX, np.array(other, dtype=np.int32), rows)
    with logitio.ForkDump(good) as a, logitio.ForkDump(pb) as b:
        rep = metrics.compare(a, b)
    assert not rep["tokens_match"] and rep["verdict"] == "FAIL"
    return "4 malformed headers rejected, token mismatch -> FAIL"


def main(argv):
    rng = np.random.default_rng(20261025)
    cases = [
        ("fork roundtrip", test_fork_roundtrip),
        ("fork verify", test_fork_verify),
        ("rt roundtrip", test_rt_roundtrip),
        ("identical dumps", test_identical),
        ("kld vs reference", test_kld_against_reference),
        ("small perturbation", test_small_perturbation_passes),
        ("16-nat floor", test_floor_and_tail),
        ("cross format align", test_cross_format_align),
        ("malformed input", test_malformed),
    ]
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in cases:
            sub = os.path.join(tmp, name.replace(" ", "-"))
            os.makedirs(sub, exist_ok=True)
            try:
                msg = fn(sub, rng)
            except AssertionError as e:
                fails += 1
                print("FAIL  %-20s %s" % (name, e))
                continue
            print("ok    %-20s %s" % (name, msg))
    print("%d/%d cases passed" % (len(cases) - fails, len(cases)))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
