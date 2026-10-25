"""Streaming readers/writers for the two logits dump formats.

fork : written by tools/perplexity/perplexity.cpp --save-all-logits (alias
       --kl-divergence-base). uint16 quantised log-probabilities, floored at
       16 nats below the row max.
rt   : the dump qwen36rt will write (spec in README.md). float32 rows, raw
       logits or log-probs, a contiguous token range.

Both formats expose the same view: a token stream plus rows, where row j
predicts the token at absolute offset + 1. Rows are handed out as
log-softmax normalised log-probabilities, so a raw-logit row and an
already-normalised log-prob row compare on the same scale.
"""

import struct

import numpy as np

FORK_MAGIC = b"_logits_"
RT_MAGIC = b"QRTLOG1\x00"
RT_VERSION = 1
NATS_FLOOR = 16.0
MAX_VOCAB = 4 << 20
MAX_CTX = 1 << 20
MAX_CHUNK = 1 << 20


class DumpError(Exception):
    """Malformed, truncated or inconsistent dump file."""


def fork_nv(n_vocab):
    """uint16 slots per position in a fork dump (2 floats + padded vocab)."""
    return 2 * ((n_vocab + 1) // 2) + 4


def log_softmax(row):
    """Normalise one row of float32 logits to log-probabilities (float32)."""
    row = np.asarray(row, dtype=np.float32)
    m = float(row.max())
    with np.errstate(over="ignore"):
        s = float(np.sum(np.exp(row - np.float32(m)), dtype=np.float64))
    return row - np.float32(m + np.log(s))


class Pos:
    """One prediction position: the row plus the token it has to predict."""

    __slots__ = ("offset", "target", "logp", "chunk", "rec")

    def __init__(self, offset, target, logp, chunk=-1, rec=-1):
        self.offset = offset
        self.target = target
        self.logp = logp
        self.chunk = chunk
        self.rec = rec


class _Dump:
    fmt = ""

    def __init__(self, path, n_vocab, n_pos, tokens):
        self.path = str(path)
        self.n_vocab = int(n_vocab)
        self.n_pos = int(n_pos)
        self.tokens = tokens

    def describe(self):
        return {
            "path": self.path,
            "fmt": self.fmt,
            "n_vocab": self.n_vocab,
            "rows": self.n_pos,
            "tokens": int(self.tokens.size),
        }

    def offsets(self):
        raise NotImplementedError

    def positions(self):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class ForkDump(_Dump):
    """Reader for tools/perplexity --save-all-logits files."""

    fmt = "fork"

    def __init__(self, path):
        self._f = open(path, "rb")
        try:
            self._read_header(path)
        except Exception:
            self._f.close()
            raise

    def _read_header(self, path):
        hdr = self._f.read(20)
        if len(hdr) < 20:
            raise DumpError("%s: shorter than the 20-byte header" % path)
        if hdr[:8] != FORK_MAGIC:
            raise DumpError("%s: magic is %r, expected %r" % (path, hdr[:8], FORK_MAGIC))
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", hdr[8:20])
        if not 2 <= n_ctx <= MAX_CTX:
            raise DumpError("%s: bad n_ctx %d" % (path, n_ctx))
        if not 1 <= n_vocab <= MAX_VOCAB:
            raise DumpError("%s: bad n_vocab %d" % (path, n_vocab))
        if not 1 <= n_chunk <= MAX_CHUNK:
            raise DumpError("%s: bad n_chunk %d" % (path, n_chunk))

        self.n_ctx = n_ctx
        self.n_chunk = n_chunk
        self.first = n_ctx // 2
        self.rec_per_chunk = n_ctx - 1 - self.first
        self.nv = fork_nv(n_vocab)
        self._row_bytes = self.nv * 2

        n_tok = n_chunk * n_ctx
        tokens = np.fromfile(self._f, dtype=np.int32, count=n_tok)
        if tokens.size != n_tok:
            raise DumpError("%s: token table truncated (%d of %d)" % (path, tokens.size, n_tok))
        super().__init__(path, n_vocab, self.rec_per_chunk * n_chunk, tokens)

        want = 20 + 4 * n_tok + self.n_pos * self._row_bytes
        got = self._f.seek(0, 2)
        if got != want:
            raise DumpError("%s: size %d, expected %d for n_ctx=%d n_vocab=%d n_chunk=%d"
                            % (path, got, want, n_ctx, n_vocab, n_chunk))
        self._f.seek(20 + 4 * n_tok)

    def offsets(self):
        if self.rec_per_chunk <= 0:
            return np.zeros(0, dtype=np.int64)
        per = (np.arange(self.n_chunk, dtype=np.int64) * self.n_ctx)[:, None]
        return (per + self.first + np.arange(self.rec_per_chunk, dtype=np.int64)[None, :]).ravel()

    def positions(self):
        buf = np.empty(self.nv, dtype=np.uint16)
        mv = memoryview(buf).cast("B")
        n_vocab = self.n_vocab
        for chunk in range(self.n_chunk):
            base = chunk * self.n_ctx + self.first
            for rec in range(self.rec_per_chunk):
                if self._f.readinto(mv) != self._row_bytes:
                    raise DumpError("%s: row data truncated at chunk %d rec %d" % (self.path, chunk, rec))
                scale, min_lp = struct.unpack("<ff", buf[:4].tobytes())
                q = buf[4:4 + n_vocab].astype(np.float32)
                logp = q * np.float32(scale) + np.float32(min_lp)
                yield Pos(base + rec, int(self.tokens[base + rec + 1]), logp, chunk, rec)

    def close(self):
        self._f.close()


class RuntimeDump(_Dump):
    """Reader for the qwen36rt float32 dump (spec in README.md)."""

    fmt = "rt"

    def __init__(self, path):
        self._f = open(path, "rb")
        self._path = str(path)
        hdr = self._f.read(24)
        if len(hdr) < 24:
            self._f.close()
            raise DumpError("%s: shorter than the 24-byte header" % path)
        if hdr[:8] != RT_MAGIC:
            self._f.close()
            raise DumpError("%s: magic is %r, expected %r" % (path, hdr[:8], RT_MAGIC))
        version, n_vocab, n_pos, start = struct.unpack("<IIII", hdr[8:24])
        if version != RT_VERSION:
            self._f.close()
            raise DumpError("%s: version %d, expected %d" % (path, version, RT_VERSION))
        if not 1 <= n_vocab <= MAX_VOCAB:
            self._f.close()
            raise DumpError("%s: bad n_vocab %d" % (path, n_vocab))
        if not 1 <= n_pos <= MAX_CTX:
            self._f.close()
            raise DumpError("%s: bad n_pos %d" % (path, n_pos))
        tokens = np.fromfile(self._f, dtype=np.int32, count=n_pos + 1)
        if tokens.size != n_pos + 1:
            self._f.close()
            raise DumpError("%s: token table truncated (%d of %d)" % (path, tokens.size, n_pos + 1))
        want = 24 + 4 * (n_pos + 1) + n_pos * n_vocab * 4
        got = self._f.seek(0, 2)
        if got != want:
            self._f.close()
            raise DumpError("%s: size %d, expected %d for n_vocab=%d n_pos=%d"
                            % (path, got, want, n_vocab, n_pos))
        self.start = int(start)
        self._row_bytes = n_vocab * 4
        self._f.seek(24 + 4 * (n_pos + 1))
        super().__init__(path, n_vocab, n_pos, tokens)

    def offsets(self):
        return self.start + np.arange(self.n_pos, dtype=np.int64)

    def positions(self):
        buf = np.empty(self.n_vocab, dtype=np.float32)
        mv = memoryview(buf).cast("B")
        for j in range(self.n_pos):
            if self._f.readinto(mv) != self._row_bytes:
                raise DumpError("%s: row data truncated at row %d" % (self.path, j))
            yield Pos(self.start + j, int(self.tokens[j + 1]), log_softmax(buf), 0, j)

    def close(self):
        self._f.close()


def open_dump(path):
    """Open a dump of either format by sniffing the magic."""
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
    except OSError as e:
        raise DumpError("cannot open %s: %s" % (path, e))
    if magic == FORK_MAGIC:
        return ForkDump(path)
    if magic == RT_MAGIC:
        return RuntimeDump(path)
    raise DumpError("%s: unknown magic %r (neither fork nor rt dump)" % (path, magic))


def quantise_row(row, n_vocab, floor=NATS_FLOOR):
    """Reproduce the fork writer: returns (scale, min_log_prob, uint16 words).

    Only used by the fixtures and by the format verifier; the runtime never
    writes the fork format.
    """
    row = np.asarray(row, dtype=np.float32)
    max_l = float(row.max())
    # the fork caps the quantised span at 16 nats below the row max
    min_l = max(float(row.min()), max_l - floor)
    s = np.sum(np.exp(row.astype(np.float64) - max_l), dtype=np.float64)
    log_sum_exp = float(np.log(s))
    min_lp = np.float32(min_l - max_l - log_sum_exp)
    scale = np.float32((max_l - min_l) / 65535.0)
    nv = fork_nv(n_vocab)
    words = np.zeros(nv, dtype=np.uint16)
    words[0:4] = np.frombuffer(np.array([scale, min_lp], dtype="<f4").tobytes(), dtype=np.uint16)
    if scale != 0:
        v = (row.astype(np.float64) - min_l) / float(scale)
        q = np.where(row > np.float32(min_l), np.rint(v), 0.0)
        words[4:4 + n_vocab] = np.clip(q, 0, 65535).astype(np.uint16)
    return scale, min_lp, words


def write_fork_dump(path, n_ctx, tokens, rows):
    """Write a fork-format dump (fixtures and format tests)."""
    tokens = np.asarray(tokens, dtype=np.int32)
    n_vocab = int(np.asarray(rows[0]).size)
    n_chunk = int(tokens.size // n_ctx)
    if tokens.size != n_chunk * n_ctx:
        raise DumpError("token count %d is not a multiple of n_ctx %d" % (tokens.size, n_ctx))
    first = n_ctx // 2
    rec = n_ctx - 1 - first
    if len(rows) != rec * n_chunk:
        raise DumpError("need %d rows for %d chunks at n_ctx %d, got %d"
                        % (rec * n_chunk, n_chunk, n_ctx, len(rows)))
    with open(path, "wb") as f:
        f.write(FORK_MAGIC)
        f.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        f.write(tokens.tobytes())
        for row in rows:
            f.write(quantise_row(row, n_vocab)[2].tobytes())


def write_runtime_dump(path, tokens, rows, start=0):
    """Write the qwen36rt float32 dump that qrt must emit later.

    tokens holds one more entry than rows: rows[j] predicts tokens[j + 1].
    """
    tokens = np.asarray(tokens, dtype=np.int32)
    r = np.asarray(rows, dtype=np.float32)
    if r.ndim != 2 or r.shape[0] + 1 != tokens.size:
        raise DumpError("rows must be [n_tokens - 1, n_vocab], got %s for %d tokens"
                        % (r.shape, tokens.size))
    with open(path, "wb") as f:
        f.write(RT_MAGIC)
        f.write(struct.pack("<IIII", RT_VERSION, r.shape[1], r.shape[0], int(start)))
        f.write(tokens.tobytes())
        f.write(r.tobytes())
