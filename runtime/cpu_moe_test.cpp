// CPU contract for the Q5_K/Q6_K expert kernels (M3-2a): each GEMV and the three
// matrix expert pass are compared with a per-element dequantising reference that
// never shares the kernel's block indexing.

#include "cpu_moe.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); \
} } while (0)

namespace {

const int QK = 256;

struct rng {
    uint32_t s;
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
    int range(int n) { return (int) (next() % (uint32_t) n); }
};

float half_to_float(uint16_t h) {
    const int sign = (h >> 15) & 1;
    const int e = (h >> 10) & 0x1f;
    int m = h & 0x3ff;
    double v;
    if (e == 0) {
        v = std::ldexp((double) m, -24);
    } else if (e == 31) {
        v = m != 0 ? NAN : INFINITY;
    } else {
        v = std::ldexp((double) (m | 0x400), e - 25);
    }
    return (float) (sign ? -v : v);
}

int block_bytes(int type) {
    switch (type) {
        case QRT_CPU_Q4K: return 144;
        case QRT_CPU_Q5K: return 176;
        default: return 210;
    }
}

void scale_min(int j, const uint8_t * q, int * sc, int * mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Half of the blocks get a subnormal delta: the real store has them and dropping
// them broke the decode path once (M3-2a).
uint16_t half_pattern(rng * r, uint16_t normal_exp) {
    if ((r->next() & 1u) != 0) {
        return (uint16_t) (normal_exp | (r->next() & 0x3ffu));
    }
    return (uint16_t) (1u | (r->next() & 0x3ffu));  // subnormal, non-zero mantissa
}

// Build one random block of the given layout. d/dmin are fp16 bit patterns, not
// values, so both sides read the same bytes.
void fill_block(int type, uint8_t * blk, rng * r) {
    const int bytes = block_bytes(type);
    memset(blk, 0, (size_t) bytes);
    if (type == QRT_CPU_Q4K) {
        const uint16_t d = half_pattern(r, 0x2000u);
        const uint16_t dm = half_pattern(r, 0x1c00u);
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dm, 2);
        for (int i = 4; i < 144; ++i) {
            blk[i] = (uint8_t) r->next();
        }
    } else if (type == QRT_CPU_Q5K) {
        const uint16_t d = half_pattern(r, 0x2000u);
        const uint16_t dm = half_pattern(r, 0x1c00u);
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dm, 2);
        for (int i = 4; i < 176; ++i) {
            blk[i] = (uint8_t) r->next();
        }
    } else {
        for (int i = 0; i < 208; ++i) {
            blk[i] = (uint8_t) r->next();
        }
        for (int i = 192; i < 208; ++i) {     // int8 scales, keep them non-zero
            blk[i] = (uint8_t) (1 + (blk[i] % 9));
        }
        const uint16_t d = half_pattern(r, 0x2800u);
        memcpy(blk + 208, &d, 2);
    }
}

// Dequantise one row (cols values) of a matrix of the given block layout.
std::vector<float> dequant_row(int type, const uint8_t * row, int cols) {
    std::vector<float> out((size_t) cols);
    const int nb = cols / QK;
    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = row + (size_t) b * block_bytes(type);
        float * base = out.data() + (size_t) b * QK;
        if (type == QRT_CPU_Q4K) {
            uint16_t dh, mh;
            memcpy(&dh, blk, 2);
            memcpy(&mh, blk + 2, 2);
            const float d = half_to_float(dh), mn = half_to_float(mh);
            const uint8_t * qs = blk + 16;
            for (int i = 0; i < QK; ++i) {
                const int j = i / 32;
                int sc, m;
                scale_min(j, blk + 4, &sc, &m);
                const int q = (qs[(j / 2) * 32 + (i % 32)] >> (4 * (j % 2))) & 0x0f;
                out[(size_t) b * QK + i] = d * (float) sc * (float) q - mn * (float) m;
            }
        } else if (type == QRT_CPU_Q5K) {
            uint16_t dh, mh;
            memcpy(&dh, blk, 2);
            memcpy(&mh, blk + 2, 2);
            const float d = half_to_float(dh), mn = half_to_float(mh);
            const uint8_t * qh = blk + 16;
            const uint8_t * qs = blk + 48;
            for (int i = 0; i < QK; ++i) {
                const int j = i / 32;
                int sc, m;
                scale_min(j, blk + 4, &sc, &m);
                int q = (qs[(j / 2) * 32 + (i % 32)] >> (4 * (j % 2))) & 0x0f;
                q += (qh[i % 32] & (1u << j)) ? 16 : 0;
                out[(size_t) b * QK + i] = d * (float) sc * (float) q - mn * (float) m;
            }
        } else {
            uint16_t dh;
            memcpy(&dh, blk + 208, 2);
            const float d = half_to_float(dh);
            const uint8_t * ql = blk;
            const uint8_t * qh = blk + 128;
            const int8_t * sc = (const int8_t *) (blk + 192);
            for (int v = 0; v < QK; ++v) {
                const int half = v / 128;
                const int local = v % 128;
                const int g = local / 32;
                const int l = local % 32;
                const uint8_t * l0 = ql + half * 64;
                const uint8_t * h0 = qh + half * 32;
                int q;
                if (g == 0) {
                    q = (l0[l] & 0x0f) | (((h0[l] >> 0) & 3) << 4);
                } else if (g == 1) {
                    q = (l0[l + 32] & 0x0f) | (((h0[l] >> 2) & 3) << 4);
                } else if (g == 2) {
                    q = (l0[l] >> 4) | (((h0[l] >> 4) & 3) << 4);
                } else {
                    q = (l0[l + 32] >> 4) | (((h0[l] >> 6) & 3) << 4);
                }
                const int s = half * 8 + g * 2 + l / 16;
                base[v] = d * (float) sc[s] * (float) (q - 32);
            }
        }
    }
    return out;
}

void gemv(int type, const void * w, int rows, int cols, const int8_t * qs,
          const float * d, const float * ssum, float * out) {
    switch (type) {
        case QRT_CPU_Q4K: qrt_gemv_q4k_q8_1(w, rows, cols, qs, d, ssum, out); break;
        case QRT_CPU_Q5K: qrt_gemv_q5k_q8_1(w, rows, cols, qs, d, ssum, out); break;
        default: qrt_gemv_q6k_q8_1(w, rows, cols, qs, d, ssum, out); break;
    }
}

void check_gemv(int type) {
    const int rows = 3;
    const int cols = 2 * QK;
    const int nb = cols / 32;
    const int bb = block_bytes(type);
    rng r{0x1234567u + (uint32_t) type * 7919u};
    std::vector<uint8_t> w((size_t) rows * (cols / QK) * bb);
    for (int rr = 0; rr < rows; ++rr) {
        for (int b = 0; b < cols / QK; ++b) {
            fill_block(type, w.data() + ((size_t) rr * (cols / QK) + b) * bb, &r);
        }
    }
    std::vector<float> x((size_t) cols);
    for (int i = 0; i < cols; ++i) {
        x[i] = (float) r.range(401) - 200.0f;
    }
    std::vector<int8_t> qs((size_t) cols);
    std::vector<float> d((size_t) nb), ssum((size_t) nb);
    qrt_q8_1_quantize(x.data(), cols, qs.data(), d.data(), ssum.data());
    std::vector<float> out((size_t) rows);
    gemv(type, w.data(), rows, cols, qs.data(), d.data(), ssum.data(), out.data());
    for (int rr = 0; rr < rows; ++rr) {
        const std::vector<float> deq = dequant_row(
            type, w.data() + (size_t) rr * (cols / QK) * bb, cols);
        double ref = 0.0;
        for (int b = 0; b < nb; ++b) {
            double part = 0.0;
            for (int i = 0; i < 32; ++i) {
                part += (double) deq[(size_t) b * 32 + i] * (double) qs[(size_t) b * 32 + i];
            }
            ref += part * (double) d[b];
        }
        const double diff = std::fabs((double) out[rr] - ref);
        CHECK(diff <= 1e-4 * (1.0 + std::fabs(ref)));
    }
}

// Reference expert pass: dequantised weights, f32 dots over the caller's Q8_1
// planes, and the same Q8_1 quantiser for the intermediate (the contract).
void ref_expert(int tg, int tu, int td, const uint8_t * wg, const uint8_t * wu,
                const uint8_t * wd, const int8_t * x_qs, const float * x_d_in,
                int n_embd, int n_ff, float * out) {
    const int nb_x = n_embd / 32;
    const int nb_y = n_ff / 32;
    std::vector<float> g((size_t) n_ff), u((size_t) n_ff), h((size_t) n_ff);
    for (int rr = 0; rr < n_ff; ++rr) {
        const std::vector<float> wgr = dequant_row(tg, wg + (size_t) rr * (n_embd / QK) * block_bytes(tg), n_embd);
        const std::vector<float> wur = dequant_row(tu, wu + (size_t) rr * (n_embd / QK) * block_bytes(tu), n_embd);
        double a = 0.0, b = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            const double qa = (double) x_qs[i] * (double) x_d_in[i / 32];
            a += (double) wgr[i] * qa;
            b += (double) wur[i] * qa;
        }
        g[rr] = (float) a;
        u[rr] = (float) b;
    }
    for (int i = 0; i < n_ff; ++i) {
        const float gg = g[i];
        h[i] = (gg / (1.0f + std::exp(-gg))) * u[i];
    }
    std::vector<int8_t> h_qs((size_t) n_ff);
    std::vector<float> h_d((size_t) nb_y), h_s((size_t) nb_y);
    qrt_q8_1_quantize(h.data(), n_ff, h_qs.data(), h_d.data(), h_s.data());
    for (int rr = 0; rr < n_embd; ++rr) {
        const std::vector<float> wdr = dequant_row(td, wd + (size_t) rr * (n_ff / QK) * block_bytes(td), n_ff);
        double a = 0.0;
        for (int i = 0; i < n_ff; ++i) {
            a += (double) wdr[i] * ((double) h_qs[i] * (double) h_d[i / 32]);
        }
        out[rr] = (float) a;
    }
}

void check_expert(int tg, int tu, int td) {
    const int n_embd = QK;
    const int n_ff = QK;
    rng r{0x9e3779b9u + (uint32_t) (tg * 131 + tu * 17 + td)};
    std::vector<uint8_t> wg((size_t) n_ff * block_bytes(tg));
    std::vector<uint8_t> wu((size_t) n_ff * block_bytes(tu));
    std::vector<uint8_t> wd((size_t) n_embd * block_bytes(td));
    for (size_t i = 0; i < wg.size() / (size_t) block_bytes(tg); ++i) {
        fill_block(tg, wg.data() + i * (size_t) block_bytes(tg), &r);
    }
    for (size_t i = 0; i < wu.size() / (size_t) block_bytes(tu); ++i) {
        fill_block(tu, wu.data() + i * (size_t) block_bytes(tu), &r);
    }
    for (size_t i = 0; i < wd.size() / (size_t) block_bytes(td); ++i) {
        fill_block(td, wd.data() + i * (size_t) block_bytes(td), &r);
    }
    std::vector<float> x((size_t) n_embd);
    for (int i = 0; i < n_embd; ++i) {
        x[i] = (float) r.range(201) - 100.0f;
    }
    std::vector<int8_t> qs((size_t) n_embd);
    std::vector<float> d((size_t) n_embd / 32), s((size_t) n_embd / 32);
    qrt_q8_1_quantize(x.data(), n_embd, qs.data(), d.data(), s.data());
    std::vector<float> got((size_t) n_embd), want((size_t) n_embd);
    qrt_cpu_moe * ctx = qrt_cpu_moe_create(4);
    CHECK(ctx != NULL);
    qrt_cpu_moe_expert_q8(ctx, tg, tu, td, wg.data(), wu.data(), wd.data(), qs.data(),
                          d.data(), s.data(), n_embd, n_ff, got.data());
    ref_expert(tg, tu, td, wg.data(), wu.data(), wd.data(), qs.data(), d.data(), n_embd,
               n_ff, want.data());
    double max_rel = 0.0;
    for (int i = 0; i < n_embd; ++i) {
        const double diff = std::fabs((double) got[i] - (double) want[i]);
        const double rel = diff / (1.0 + std::fabs((double) want[i]));
        if (rel > max_rel) {
            max_rel = rel;
        }
    }
    CHECK(max_rel <= 1e-4);

    // The float wrapper quantises the same input planes and must land on the same row.
    std::vector<float> via_float((size_t) n_embd);
    qrt_cpu_moe_expert_typed(ctx, tg, tu, td, wg.data(), wu.data(), wd.data(), x.data(),
                             n_embd, n_ff, via_float.data());
    for (int i = 0; i < n_embd; ++i) {
        CHECK(std::fabs(via_float[i] - got[i]) <= 1e-4 * (1.0f + std::fabs(got[i])));
    }
    // The Q4_K entry point stays the 0/0/0 case.
    if (tg == QRT_CPU_Q4K && tu == QRT_CPU_Q4K && td == QRT_CPU_Q4K) {
        std::vector<float> legacy((size_t) n_embd);
        qrt_cpu_moe_expert(ctx, wg.data(), wu.data(), wd.data(), x.data(), n_embd, n_ff,
                           legacy.data());
        for (int i = 0; i < n_embd; ++i) {
            CHECK(legacy[i] == via_float[i]);
        }
    }
    qrt_cpu_moe_free(ctx);
}

} // namespace

int main() {
    check_gemv(QRT_CPU_Q4K);
    check_gemv(QRT_CPU_Q5K);
    check_gemv(QRT_CPU_Q6K);
    check_expert(QRT_CPU_Q4K, QRT_CPU_Q4K, QRT_CPU_Q4K);
    check_expert(QRT_CPU_Q4K, QRT_CPU_Q4K, QRT_CPU_Q5K);
    check_expert(QRT_CPU_Q4K, QRT_CPU_Q4K, QRT_CPU_Q6K);
    check_expert(QRT_CPU_Q5K, QRT_CPU_Q5K, QRT_CPU_Q6K);
    std::printf("cpu_moe_contract PASS kernels=q4k,q5k,q6k experts=4\n");
    return 0;
}
