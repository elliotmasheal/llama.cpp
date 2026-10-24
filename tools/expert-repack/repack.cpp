// expert-repack: writes <model>.qwen36rt.bin + .idx per runtime/repack/FORMAT.md (M5-2).
// Copies every expert slice byte for byte, hashes it, and journals each record so an
// interrupted run can resume. No quant value is modified.

#include "gguf_index.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

static const uint32_t ALIGN = 65536;
static const char     MAGIC[8] = {'Q', '3', '6', 'R', 'T', '0', '1', '\0'};

// ---------------------------------------------------------------- sha256

struct Sha256 {
    uint32_t h[8];
    uint64_t len;
    uint8_t  buf[64];
    size_t   n;
};

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha_block(Sha256 * s, const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 |
               (uint32_t) p[4 * i + 2] << 8 | (uint32_t) p[4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3];
    uint32_t e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + S1 + ch + K256[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
    s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void sha_init(Sha256 * s) {
    static const uint32_t H0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, H0, sizeof(H0));
    s->len = 0;
    s->n = 0;
}

static void sha_update(Sha256 * s, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    s->len += n;
    while (n > 0) {
        const size_t take = (64 - s->n) < n ? (64 - s->n) : n;
        memcpy(s->buf + s->n, p, take);
        s->n += take;
        p += take;
        n -= take;
        if (s->n == 64) {
            sha_block(s, s->buf);
            s->n = 0;
        }
    }
}

static void sha_final(Sha256 * s, uint8_t out[32]) {
    const uint64_t bits = s->len * 8;
    const uint8_t one = 0x80;
    sha_update(s, &one, 1);
    const uint8_t zero = 0;
    while (s->n != 56) {
        sha_update(s, &zero, 1);
    }
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) {
        len[7 - i] = (uint8_t) (bits >> (8 * i));
    }
    sha_update(s, len, 8);
    for (int i = 0; i < 8; ++i) {
        out[4 * i]     = (uint8_t) (s->h[i] >> 24);
        out[4 * i + 1] = (uint8_t) (s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t) (s->h[i] >> 8);
        out[4 * i + 3] = (uint8_t) (s->h[i]);
    }
}

static std::string hex(const uint8_t * p, size_t n) {
    static const char * D = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        s.push_back(D[p[i] >> 4]);
        s.push_back(D[p[i] & 15]);
    }
    return s;
}

#ifdef _WIN32
#define XSEEK _fseeki64
#else
#define XSEEK fseeko
#endif

// ---------------------------------------------------------------- repack

struct Args {
    const char * gguf = nullptr;
    const char * out = nullptr;
    uint32_t limit_layers = 0;   // 0 = all
    uint32_t limit_experts = 0;  // 0 = all
    bool verify = false;
    bool resume = false;
};

static void usage() {
    std::printf("usage: repack --gguf PATH --out PREFIX [--limit-layers N] [--limit-experts N]\n"
                "               [--resume] [--verify]\n");
}

// Reads [off, off+bytes) from f into buf and feeds it to the hash.
static int copy_hashed(FILE * fin, uint64_t off, uint64_t bytes, std::vector<uint8_t> & block,
                       Sha256 * sha, size_t cap) {
    if (XSEEK(fin, (long long) off, SEEK_SET) != 0) {
        std::fprintf(stderr, "repack: seek to %llu failed\n", (unsigned long long) off);
        return -1;
    }
    std::vector<uint8_t> tmp(cap);
    uint64_t left = bytes;
    while (left > 0) {
        const size_t take = (size_t) (left < cap ? left : cap);
        if (fread(tmp.data(), 1, take, fin) != take) {
            std::fprintf(stderr, "repack: short read at %llu\n", (unsigned long long) off);
            return -1;
        }
        sha_update(sha, tmp.data(), take);
        block.insert(block.end(), tmp.begin(), tmp.begin() + take);
        left -= take;
    }
    return 0;
}

static int write_all(FILE * f, const void * p, size_t n) {
    return fwrite(p, 1, n, f) == n ? 0 : -1;
}

static int verify_index(const char * idx_path, const char * bin_path);

int main(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        if (f == "--gguf" && i + 1 < argc) a.gguf = argv[++i];
        else if (f == "--out" && i + 1 < argc) a.out = argv[++i];
        else if (f == "--limit-layers" && i + 1 < argc) a.limit_layers = (uint32_t) atoi(argv[++i]);
        else if (f == "--limit-experts" && i + 1 < argc) a.limit_experts = (uint32_t) atoi(argv[++i]);
        else if (f == "--verify") a.verify = true;
        else if (f == "--resume") a.resume = true;
        else { usage(); return 2; }
    }
    if (a.gguf == nullptr || a.out == nullptr) {
        usage();
        return 2;
    }
    const std::string bin_path = std::string(a.out) + ".qwen36rt.bin";
    const std::string idx_path = std::string(a.out) + ".qwen36rt.idx";
    if (a.verify) {
        return verify_index(idx_path.c_str(), bin_path.c_str()) == 0 ? 0 : 1;
    }

    qrt_gguf * g = nullptr;
    char err[256] = {0};
    if (qrt_gguf_open(a.gguf, &g, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "repack: %s\n", err);
        return 2;
    }
    const qrt_gguf_hparams * hp = qrt_gguf_hparams_of(g);
    if (!(hp->present & QRT_HP_BLOCK_COUNT) || !(hp->present & QRT_HP_EXPERT_COUNT)) {
        std::fprintf(stderr, "repack: metadata lacks block_count/expert_count\n");
        qrt_gguf_close(g);
        return 2;
    }
    const uint32_t layers  = (uint32_t) hp->block_count;
    const uint32_t experts = (uint32_t) hp->expert_count;
    const uint32_t n_layer = a.limit_layers ? a.limit_layers : layers;
    const uint32_t n_exp   = a.limit_experts ? a.limit_experts : experts;
    if (n_layer > layers || n_exp > experts) {
        std::fprintf(stderr, "repack: limit above the file (%u layers, %u experts)\n", layers, experts);
        qrt_gguf_close(g);
        return 2;
    }

    const bool partial = a.resume;
    auto exists = [](const std::string & p) {
        FILE * f = fopen(p.c_str(), "rb");
        if (f != nullptr) {
            fclose(f);
            return true;
        }
        return false;
    };
    // Resume prefers the .partial pair; a finished pair is appended to directly.
    const bool resume_finals = partial && !exists(idx_path + ".partial") && exists(idx_path);
    const std::string bin_tmp = resume_finals ? bin_path : bin_path + (partial ? ".partial" : ".building");
    const std::string idx_tmp = resume_finals ? idx_path : idx_path + (partial ? ".partial" : ".building");
    FILE * fin = fopen(a.gguf, "rb");
    if (fin == nullptr) {
        std::fprintf(stderr, "repack: cannot open %s\n", a.gguf);
        qrt_gguf_close(g);
        return 2;
    }
    FILE * fb = fopen(bin_tmp.c_str(), partial ? "ab+" : "wb+");
    FILE * fi = fopen(idx_tmp.c_str(), partial ? "a" : "w");
    if (fb == nullptr || fi == nullptr) {
        std::fprintf(stderr, "repack: cannot open outputs\n");
        fclose(fin);
        qrt_gguf_close(g);
        return 2;
    }
    uint64_t payload = 0;
    uint32_t last_layer = 0, last_expert = 0;
    bool have_last = false;
    if (partial) {
        FILE * fr = fopen(idx_tmp.c_str(), "r");
        if (fr != nullptr) {
            char ln[512];
            while (fgets(ln, sizeof(ln), fr) != nullptr) {
                unsigned L2 = 0, E2 = 0;
                unsigned long long o2 = 0, b2 = 0, g2 = 0, u2 = 0, d2 = 0;
                char h2[65] = {0};
                if (ln[0] != '#' &&
                    sscanf(ln, "%u,%u,%llu,%llu,%llu,%llu,%llu,%64s",
                           &L2, &E2, &o2, &b2, &g2, &u2, &d2, h2) == 8) {
                    last_layer = L2;
                    last_expert = E2;
                    payload = o2 + b2;
                    have_last = true;
                }
            }
            fclose(fr);
        }
        if (!have_last) {
            std::fprintf(stderr, "repack: --resume but %s has no complete record\n", idx_tmp.c_str());
            return 2;
        }
        if (XSEEK(fb, 0, SEEK_END) != 0) {
            std::fprintf(stderr, "repack: cannot seek to the end for --resume\n");
            return 2;
        }
    }
    if (!partial) {
        uint8_t hdr[64];
        memset(hdr, 0, sizeof(hdr));
        memcpy(hdr, MAGIC, 8);
        const uint32_t v = 1, al = ALIGN;
        memcpy(hdr + 8, &v, 4);
        memcpy(hdr + 12, &al, 4);
        memcpy(hdr + 16, &layers, 4);
        memcpy(hdr + 20, &experts, 4);
        const uint64_t poff = 64, n_tot = (uint64_t) layers * experts;
        memcpy(hdr + 24, &poff, 8);
        memcpy(hdr + 40, &n_tot, 8);
        if (write_all(fb, hdr, sizeof(hdr)) != 0) {
            std::fprintf(stderr, "repack: header write failed\n");
            return 2;
        }
        payload = 0;
        std::fprintf(fi, "# qwen36rt index v1\n");
        std::fprintf(fi, "# gguf=%s\n# gguf_size=%llu\n", a.gguf,
                     (unsigned long long) qrt_gguf_file_size(g));
        std::fprintf(fi, "# arch=%s\n# layers=%u experts=%u align=%u\n",
                     hp->arch, layers, experts, ALIGN);
        std::fprintf(fi, "# columns=layer,expert,offset,bytes,gate,up,down,sha256\n");
    }

    const uint32_t start_layer = 0;
    uint32_t done = 0, failed = 0;
    const size_t cap = 1u << 20;
    for (uint32_t L = start_layer; L < n_layer; ++L) {
        char name[64];
        const qrt_gguf_tensor * t[3] = {nullptr, nullptr, nullptr};
        const char * part[3] = {"gate", "up", "down"};
        for (int p = 0; p < 3; ++p) {
            snprintf(name, sizeof(name), "blk.%u.ffn_%s_exps.weight", L, part[p]);
            t[p] = qrt_gguf_find(g, name);
            if (t[p] == nullptr || t[p]->n_dims < 3 || t[p]->ne[2] != experts ||
                (t[p]->nbytes % experts) != 0) {
                std::fprintf(stderr, "repack: %s is missing or not expert-major\n", name);
                ++failed;
                continue;
            }
        }
        if (failed) {
            break;
        }
        for (uint32_t e = 0; e < n_exp; ++e) {
            if (partial && (L < last_layer || (L == last_layer && e <= last_expert))) {
                continue;
            }
            std::vector<uint8_t> block;
            Sha256 sha;
            sha_init(&sha);
            uint64_t sizes[3];
            for (int p = 0; p < 3; ++p) {
                const uint64_t stride = t[p]->nbytes / experts;
                sizes[p] = stride;
                if (copy_hashed(fin, t[p]->offset + (uint64_t) e * stride, stride, block, &sha, cap)) {
                    ++failed;
                    break;
                }
            }
            if (failed) {
                break;
            }
            const uint64_t pad = payload % ALIGN == 0 ? 0 : ALIGN - payload % ALIGN;
            if (pad) {
                std::vector<uint8_t> z((size_t) pad, 0);
                if (write_all(fb, z.data(), z.size()) != 0) { ++failed; break; }
            }
            const uint64_t off = payload + pad;
            if (write_all(fb, block.data(), block.size()) != 0) { ++failed; break; }
            uint8_t digest[32];
            sha_final(&sha, digest);
            std::fprintf(fi, "%u,%u,%llu,%llu,%llu,%llu,%llu,%s\n", L, e,
                         (unsigned long long) off, (unsigned long long) block.size(),
                         (unsigned long long) sizes[0], (unsigned long long) sizes[1],
                         (unsigned long long) sizes[2], hex(digest, 32).c_str());
            payload = off + block.size();
            ++done;
        }
        std::fflush(fi);
        if (failed) {
            break;
        }
    }
    {
        // Whole-payload SHA-256: one extra pass over the bytes just written.
        Sha256 whole;
        sha_init(&whole);
        std::vector<uint8_t> buf(1u << 20);
        if (XSEEK(fb, 64, SEEK_SET) == 0) {
            size_t got = 0;
            while ((got = fread(buf.data(), 1, buf.size(), fb)) > 0) {
                sha_update(&whole, buf.data(), got);
            }
            uint8_t dg[32];
            sha_final(&whole, dg);
            std::fprintf(fi, "# payload_sha256=%s\n", hex(dg, 32).c_str());
        }
    }
    fflush(fb);
    fclose(fb);
    fclose(fi);
    fclose(fin);
    if (failed) {
        std::fprintf(stderr, "repack: stopped after %u experts; %s kept for --resume\n", done, bin_tmp.c_str());
        qrt_gguf_close(g);
        return 1;
    }
    std::printf("repack: experts=%u payload_bytes=%llu bin=%s idx=%s\n", done,
                (unsigned long long) payload, bin_tmp.c_str(), idx_tmp.c_str());
    // The index is renamed last: its presence means the pair is complete.
    if (!resume_finals) {
        remove(bin_path.c_str());
        remove(idx_path.c_str());
        if (std::rename(bin_tmp.c_str(), bin_path.c_str()) != 0 ||
            std::rename(idx_tmp.c_str(), idx_path.c_str()) != 0) {
            std::fprintf(stderr, "repack: rename of the finished pair failed\n");
            qrt_gguf_close(g);
            return 2;
        }
    }
    qrt_gguf_close(g);
    return 0;
}

static int verify_index(const char * idx_path, const char * bin_path) {
    FILE * fi = fopen(idx_path, "r");
    FILE * fb = fopen(bin_path, "rb");
    if (fi == nullptr || fb == nullptr) {
        std::fprintf(stderr, "verify: cannot open %s / %s\n", idx_path, bin_path);
        if (fi) fclose(fi);
        if (fb) fclose(fb);
        return -1;
    }
    char line[512];
    uint32_t checked = 0, bad = 0;
    while (fgets(line, sizeof(line), fi) != nullptr) {
        if (line[0] == '#') {
            continue;
        }
        unsigned L, E;
        unsigned long long off, bytes, g, u, d;
        char want[65] = {0};
        if (sscanf(line, "%u,%u,%llu,%llu,%llu,%llu,%llu,%64s", &L, &E, &off, &bytes, &g, &u, &d, want) != 8) {
            continue;
        }
        if (XSEEK(fb, (long long) (64 + off), SEEK_SET) != 0) {
            ++bad;
            continue;
        }
        Sha256 sha;
        sha_init(&sha);
        std::vector<uint8_t> buf(1u << 20);
        unsigned long long left = bytes;
        while (left > 0) {
            const size_t take = (size_t) (left < buf.size() ? left : buf.size());
            if (fread(buf.data(), 1, take, fb) != take) {
                ++bad;
                break;
            }
            sha_update(&sha, buf.data(), take);
            left -= take;
        }
        uint8_t digest[32];
        sha_final(&sha, digest);
        if (hex(digest, 32) != want) {
            std::fprintf(stderr, "verify: mismatch at layer %u expert %u\n", L, E);
            ++bad;
        }
        ++checked;
    }
    fclose(fi);
    fclose(fb);
    std::printf("verify: checked=%u bad=%u result=%s\n", checked, bad, bad == 0 ? "pass" : "fail");
    return bad == 0 ? 0 : -1;
}
