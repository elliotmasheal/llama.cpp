#include "kernels/moe_expert.h"

#include "kernels/gemm.h"
#include "weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef QRT_HAVE_VULKAN

namespace {

const uint32_t HIDDEN = 2048;
const uint32_t FFN    = 512;

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_size, fmt, ap);
    va_end(ap);
    return -1;
}

// Deterministic Q8_1 activations for the harness. The real path quantizes the layer
// input; until that exists, a fixed pattern keeps two runs comparable.
uint32_t lcg(uint32_t * s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

void fill_acts(uint32_t tokens, uint32_t hidden, unsigned char * qs, uint32_t * ds) {
    uint32_t seed = 12345u;
    const uint32_t blocks = tokens * (hidden / 32);
    for (uint32_t b = 0; b < blocks; ++b) {
        unsigned char * q = qs + (size_t) b * 32;
        int32_t sum = 0;
        for (uint32_t j = 0; j < 32; ++j) {
            const int32_t v = (int32_t) (lcg(&seed) % 255u) - 127;
            q[j] = (unsigned char) (int8_t) v;
            sum += v;
        }
        // Delta around 1e-2 keeps the products inside the fp32 range of the kernel.
        const uint32_t d = 0x2800u + (lcg(&seed) & 0x0FFFu);   // f16 ~0.0098..0.0195
        ds[b] = d | ((uint32_t) (uint16_t) (sum + 32768) << 16);
    }
}

uint32_t f16_to_f32_bits(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    if (exp == 0) {
        if (man == 0) {
            return sign;
        }
        uint32_t e = 127 - 15 + 1;
        uint32_t m = man;
        while ((m & 0x400u) == 0) {
            m <<= 1;
            --e;
        }
        return sign | (e << 23) | ((m & 0x3FFu) << 13);
    }
    if (exp == 31) {
        return sign | 0x7F800000u | (man << 13);
    }
    return sign | ((exp + 127 - 15) << 23) | (man << 13);
}

float f16_to_f32(uint32_t h) {
    const uint32_t bits = f16_to_f32_bits(h & 0xFFFFu);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

} // namespace

struct qrt_moe {
    qrt_vk_dev *  dev;
    qrt_weights * w;
    qrt_moe_cfg   cfg;
    qrt_gemm *     gemm;
    uint32_t      down_skipped;   // experts whose down matrix is not Q4_K yet
    uint32_t      experts_run;
    VkBuffer      a_qs;
    VkDeviceMemory a_qs_mem;
    VkBuffer      a_ds;
    VkDeviceMemory a_ds_mem;
    VkBuffer      h_qs;
    VkDeviceMemory h_qs_mem;
    VkBuffer      h_ds;
    VkDeviceMemory h_ds_mem;
    VkBuffer      g_out;
    VkDeviceMemory g_out_mem;
    VkBuffer      u_out;
    VkDeviceMemory u_out_mem;
    VkBuffer      o_out;
    VkDeviceMemory o_out_mem;
};

static int make_dev_buffer(qrt_vk_dev * dev, uint64_t bytes, VkBuffer * buf,
                           VkDeviceMemory * mem, char * err, size_t err_size) {
    return qrt_vk_alloc(dev, bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                        buf, mem, err, err_size);
}

int qrt_moe_open(qrt_vk_dev * dev, qrt_weights * w, const char * spv_dir,
                 const qrt_moe_cfg * cfg, qrt_moe ** out, char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || w == NULL || cfg == NULL || spv_dir == NULL) {
        return fail(err, err_size, "the MoE leg needs a device, a store and a shader dir");
    }
    if (cfg->tokens == 0 || cfg->tokens % 32 != 0) {
        return fail(err, err_size, "tokens per expert must be a non-zero multiple of 32");
    }
    qrt_moe * m = (qrt_moe *) calloc(1, sizeof(qrt_moe));
    if (m == NULL) {
        return fail(err, err_size, "out of memory for the MoE leg");
    }
    m->dev = dev;
    m->w   = w;
    m->cfg = *cfg;
    if (qrt_gemm_open(dev, spv_dir, &m->gemm, err, err_size) != 0) {
        qrt_moe_close(dev, m);
        return -1;
    }
    const uint32_t t = cfg->tokens;
    const uint64_t a_qs_bytes = (uint64_t) t * cfg->hidden;
    const uint64_t a_ds_bytes = (uint64_t) t * (cfg->hidden / 32) * 4;
    const uint64_t h_qs_bytes = (uint64_t) t * cfg->ffn;
    const uint64_t h_ds_bytes = (uint64_t) t * (cfg->ffn / 32) * 4;
    const uint64_t g_bytes    = (uint64_t) t * cfg->ffn * 4;
    const uint64_t o_bytes    = (uint64_t) t * cfg->hidden * 4;
    const struct {
        uint64_t bytes;
        VkBuffer * buf;
        VkDeviceMemory * mem;
        const char * what;
    } slots[] = {
        { a_qs_bytes, &m->a_qs, &m->a_qs_mem, "input qs planes" },
        { a_ds_bytes, &m->a_ds, &m->a_ds_mem, "input ds plane" },
        { h_qs_bytes, &m->h_qs, &m->h_qs_mem, "hidden qs planes" },
        { h_ds_bytes, &m->h_ds, &m->h_ds_mem, "hidden ds plane" },
        { g_bytes,    &m->g_out, &m->g_out_mem, "gate output" },
        { g_bytes,    &m->u_out, &m->u_out_mem, "up output" },
        { o_bytes,    &m->o_out, &m->o_out_mem, "expert output" },
    };
    for (size_t i = 0; i < sizeof(slots) / sizeof(slots[0]); ++i) {
        // The plane buffers are rounded up to the allocation granularity the driver asks
        // for; qrt_vk_alloc handles that, the small ones would otherwise be 0 bytes.
        const uint64_t bytes = slots[i].bytes < 4096 ? 4096 : slots[i].bytes;
        if (make_dev_buffer(dev, bytes, slots[i].buf, slots[i].mem, err, err_size) != 0) {
            qrt_moe_close(dev, m);
            return -1;
        }
    }

    unsigned char * host_qs = (unsigned char *) malloc((size_t) a_qs_bytes);
    uint32_t * host_ds = (uint32_t *) malloc((size_t) a_ds_bytes);
    if (host_qs == NULL || host_ds == NULL) {
        free(host_qs);
        free(host_ds);
        qrt_moe_close(dev, m);
        return fail(err, err_size, "out of memory for the activation planes");
    }
    fill_acts(t, cfg->hidden, host_qs, host_ds);
    qrt_vk_staging * stg = NULL;
    int rc = 0;
    if (qrt_vk_staging_create(dev, a_qs_bytes, &stg, err, err_size) != 0) {
        rc = -1;
    } else if (qrt_vk_upload(dev, stg, m->a_qs, 0, host_qs, a_qs_bytes, err, err_size) != 0 ||
               qrt_vk_upload(dev, stg, m->a_ds, 0, host_ds, a_ds_bytes, err, err_size) != 0) {
        rc = -1;
    }
    qrt_vk_staging_destroy(dev, stg);
    free(host_qs);
    free(host_ds);
    if (rc != 0) {
        qrt_moe_close(dev, m);
        return -1;
    }
    *out = m;
    return 0;
}

void qrt_moe_close(qrt_vk_dev * dev, qrt_moe * m) {
    if (m == NULL) {
        return;
    }
    qrt_gemm_close(dev, m->gemm);
    if (dev != NULL) {
        const struct {
            VkBuffer * buf;
            VkDeviceMemory * mem;
        } bufs[] = {
            { &m->a_qs, &m->a_qs_mem }, { &m->a_ds, &m->a_ds_mem },
            { &m->h_qs, &m->h_qs_mem }, { &m->h_ds, &m->h_ds_mem },
            { &m->g_out, &m->g_out_mem }, { &m->u_out, &m->u_out_mem },
            { &m->o_out, &m->o_out_mem },
        };
        for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); ++i) {
            if (*bufs[i].mem != VK_NULL_HANDLE) {
                qrt_vk_free(dev, *bufs[i].buf, *bufs[i].mem);
            }
        }
    }
    free(m);
}

uint32_t qrt_moe_tokens(const qrt_moe * m) { return m != NULL ? m->cfg.tokens : 0; }

uint32_t qrt_moe_down_skipped(const qrt_moe * m) {
    return m != NULL ? m->down_skipped : 0;
}

uint32_t qrt_moe_experts_run(const qrt_moe * m) { return m != NULL ? m->experts_run : 0; }

uint64_t qrt_moe_flops(const qrt_moe * m) {
    if (m == NULL) {
        return 0;
    }
    // gate, up and down, each 2 x n x k multiply-adds per token.
    const uint64_t n = m->cfg.tokens;
    return 3ull * 2ull * n * m->cfg.ffn * m->cfg.hidden;
}

int qrt_moe_compute(void * user, uint32_t layer, uint32_t expert, VkBuffer buf,
                    uint64_t slot_offset, uint64_t bytes, char * err, size_t err_size) {
    qrt_moe * m = (qrt_moe *) user;
    if (m == NULL || buf == VK_NULL_HANDLE) {
        return fail(err, err_size, "the MoE leg has no slot buffer");
    }
    qrt_expert_loc loc;
    if (qrt_weights_locate(m->w, layer, expert, &loc, err, err_size) != 0) {
        return -1;
    }
    if (slot_offset + loc.bytes > bytes + slot_offset) {
        return fail(err, err_size, "expert %u,%u does not fit the slot", layer, expert);
    }
    const uint32_t t = m->cfg.tokens;
    const uint32_t hidden = m->cfg.hidden;
    const uint32_t ffn = m->cfg.ffn;

    qrt_gemm_args gate;
    memset(&gate, 0, sizeof(gate));
    gate.w = buf;
    gate.a_qs = m->a_qs;
    gate.a_ds = m->a_ds;
    gate.out = m->g_out;
    gate.n = ffn;
    gate.k = hidden;
    gate.m = t;

    qrt_gemm_args up = gate;
    up.out = m->u_out;

    qrt_gemm_args down = gate;
    down.a_qs = m->h_qs;
    down.a_ds = m->h_ds;
    down.out = m->o_out;
    down.n = hidden;
    down.k = ffn;

    // The store mixes quant types per matrix (FORMAT.md): gate/up Q4_K on most layers,
    // down Q5_K, layers 34/38 Q6_K down, layer 39 Q5_K gate/up. The type is derived from
    // the slice size, and an unknown size is a counted error rather than a misread.
    const int t_gate = qrt_gemm_type_of(ffn, hidden, loc.gate);
    const int t_up = qrt_gemm_type_of(ffn, hidden, loc.up);
    const int t_down = qrt_gemm_type_of(hidden, ffn, loc.down);
    if (t_gate < 0 || t_up < 0) {
        return fail(err, err_size,
                    "expert %u,%u gate/up of %llu/%llu bytes matches no known type", layer,
                    expert, (unsigned long long) loc.gate, (unsigned long long) loc.up);
    }
    if (t_down < 0) {
        return fail(err, err_size, "expert %u,%u down of %llu bytes matches no known type",
                    layer, expert, (unsigned long long) loc.down);
    }
    gate.type = (qrt_quant) t_gate;
    up.type   = (qrt_quant) t_up;
    down.type = (qrt_quant) t_down;
    // Q4_K and Q5_K address the weight buffer in words, Q6_K in bytes (210-byte blocks).
    const uint32_t off_gate = (uint32_t) slot_offset;
    const uint32_t off_up   = (uint32_t) (slot_offset + loc.gate);
    const uint32_t off_down = (uint32_t) (slot_offset + loc.gate + loc.up);
    gate.wbase = gate.type == QRT_QUANT_Q6K ? off_gate : off_gate / 4;
    up.wbase   = up.type == QRT_QUANT_Q6K ? off_up : off_up / 4;
    down.wbase = down.type == QRT_QUANT_Q6K ? off_down : off_down / 4;

    // One command buffer for all dispatches: at 64 tokens each GEMM is ~0.03 ms of math,
    // so four submit+fence rounds would cost more than the work (RESULTS 39).
    if (qrt_gemm_begin(m->gemm, err, err_size) != 0 ||
        qrt_gemm_record(m->gemm, &gate, err, err_size) != 0 ||
        qrt_gemm_record(m->gemm, &up, err, err_size) != 0 ||
        qrt_gemm_record_swiglu(m->gemm, m->g_out, m->u_out, m->h_qs, m->h_ds, t, ffn,
                              err, err_size) != 0 ||
        qrt_gemm_record(m->gemm, &down, err, err_size) != 0) {
        return -1;
    }
    if (qrt_gemm_submit(m->gemm, err, err_size) != 0) {
        return -1;
    }
    ++m->experts_run;
    return 0;
}

// Canonical dequant of the three K-quants (ggml dequantize_row_q*_K order) times the
// Q8_1 activation, in double, folded exactly like the kernels claim to fold it.
static void ref_gemm_row(const unsigned char * w, qrt_quant type, const unsigned char * a_qs,
                         const uint32_t * a_ds, uint32_t n_row, uint32_t k, uint32_t m_tok,
                         float * out) {
    const size_t block = qrt_gemm_block_bytes(type);
    const uint32_t nsb = k / 256;
    const uint32_t nk32 = k / 32;
    const unsigned char * wrow = w + (size_t) n_row * nsb * block;
    for (uint32_t tok = 0; tok < m_tok; ++tok) {
        double acc = 0.0;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const unsigned char * blk = wrow + (size_t) sb * block;
            if (type == QRT_QUANT_Q6K) {
                const float d = f16_to_f32((uint32_t) blk[208] | ((uint32_t) blk[209] << 8));
                for (uint32_t hh = 0; hh < 2; ++hh) {
                    for (uint32_t s = 0; s < 4; ++s) {
                        const uint32_t abi = tok * nk32 + sb * 8 + hh * 4 + s;
                        const float aq = f16_to_f32(a_ds[abi] & 0xFFFFu);
                        for (uint32_t i = 0; i < 8; ++i) {
                            const uint32_t l = 4 * i;
                            uint32_t ql = 0;
                            uint32_t qh = 0;
                            for (uint32_t b = 0; b < 4; ++b) {
                                ql |= (uint32_t) blk[hh * 64 + ((s & 1u) ? 32 : 0) + l + b] << (8 * b);
                                qh |= (uint32_t) blk[128 + hh * 32 + l + b] << (8 * b);
                            }
                            const uint32_t q4 = (s < 2) ? (ql & 0x0F0F0F0Fu)
                                                        : ((ql >> 4) & 0x0F0F0F0Fu);
                            const uint32_t h2 = (qh >> (2 * s)) & 0x03030303u;
                            const uint32_t q6 = (q4 | (h2 << 4)) ^ 0x20202020u;
                            const int sc_idx = (int) (hh * 8 + 2 * s + (i >= 4 ? 1 : 0));
                            int8_t sv = (int8_t) blk[192 + sc_idx];
                            for (uint32_t e = 0; e < 4; ++e) {
                                const int8_t qv = (int8_t) (uint8_t) ((q6 >> (8 * e)) & 0xFFu);
                                const int8_t qa = (int8_t) a_qs[(size_t) abi * 32 + l + e];
                                acc += (double) d * (double) sv * (double) qv *
                                       ((double) aq * (double) qa);
                            }
                        }
                    }
                }
                continue;
            }
            const float d = f16_to_f32((uint32_t) blk[0] | ((uint32_t) blk[1] << 8));
            const float dmin = f16_to_f32((uint32_t) blk[2] | ((uint32_t) blk[3] << 8));
            const unsigned char * scales = blk + 4;
            const unsigned char * qh = type == QRT_QUANT_Q5K ? blk + 16 : NULL;
            const unsigned char * qs = blk + (type == QRT_QUANT_Q5K ? 48 : 16);
            for (uint32_t g = 0; g < 4; ++g) {
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t j = 2 * g + half;
                    uint32_t scv;
                    uint32_t mnv;
                    if (j < 4) {
                        scv = scales[j] & 63u;
                        mnv = scales[j + 4] & 63u;
                    } else {
                        scv = (scales[j + 4] & 15u) | ((scales[j - 4] >> 6u) << 4);
                        mnv = (scales[j + 4] >> 4u) | ((scales[j] >> 6u) << 4);
                    }
                    const uint32_t abi = tok * nk32 + sb * 8 + j;
                    const float aq = f16_to_f32(a_ds[abi] & 0xFFFFu);
                    double sub = 0.0;
                    for (uint32_t e = 0; e < 32; ++e) {
                        const uint8_t byte = qs[32 * g + e];
                        uint32_t q = half == 0 ? (byte & 15u) : (byte >> 4u);
                        if (qh != NULL) {
                            q |= ((qh[e] >> j) & 1u) << 4;
                        }
                        const int8_t qa = (int8_t) a_qs[(size_t) abi * 32 + e];
                        sub += ((double) d * (double) scv * (double) q -
                                (double) dmin * (double) mnv) * ((double) aq * (double) qa);
                    }
                    acc += sub;
                }
            }
        }
        out[tok] = (float) acc;
    }
}

struct check_arm {
    qrt_gemm_args args;
    const unsigned char * host;     // host copy of the matrix block
    uint64_t bytes;
    uint32_t rows;
    qrt_quant type;
    float * gpu;
    float * want;
};

// Runs one matrix of the expert and compares it against the reference. Rotation of the
// output buffer and the scratch planes is shared with qrt_moe_compute.
static int check_one(qrt_vk_dev * dev, qrt_moe * m, check_arm * arm, const unsigned char * a_qs,
                     const uint32_t * a_ds, double * rel_l1, char * err, size_t err_size) {
    const uint32_t t = m->cfg.tokens;
    if (qrt_gemm_dispatch(m->gemm, &arm->args, err, err_size) != 0) {
        return -1;
    }
    qrt_vk_staging * stg = NULL;
    if (qrt_vk_staging_create(dev, (uint64_t) t * arm->args.n * 4, &stg, err, err_size) != 0) {
        return -1;
    }
    int rc = -1;
    if (qrt_vk_download(dev, stg, arm->args.out, (uint64_t) arm->args.obase * 4, arm->gpu,
                        (uint64_t) t * arm->args.n * 4, err, err_size) == 0) {
        for (uint32_t row = 0; row < arm->rows; ++row) {
            ref_gemm_row(arm->host, arm->type, a_qs, a_ds, row, arm->args.k, t,
                         arm->want + (size_t) row * t);
        }
        double num = 0.0;
        double den = 0.0;
        for (uint32_t row = 0; row < arm->rows; ++row) {
            for (uint32_t tok = 0; tok < t; ++tok) {
                const double a = arm->gpu[(size_t) row * t + tok];
                const double b = arm->want[(size_t) row * t + tok];
                num += fabs(a - b);
                den += fabs(b);
            }
        }
        *rel_l1 = den > 0.0 ? num / den : num;
        rc = 0;
    }
    qrt_vk_staging_destroy(dev, stg);
    return rc;
}

int qrt_moe_check(qrt_vk_dev * dev, qrt_moe * m, qrt_weights * w, uint32_t layer,
                  uint32_t expert, uint32_t rows, qrt_moe_check_result * out, char * err,
                  size_t err_size) {
    if (m == NULL || w == NULL || out == NULL) {
        return fail(err, err_size, "the check needs the MoE leg and the store");
    }
    memset(out, 0, sizeof(*out));
    qrt_expert_loc loc;
    if (qrt_weights_locate(w, layer, expert, &loc, err, err_size) != 0) {
        return -1;
    }
    const uint32_t t = m->cfg.tokens;
    const uint32_t hidden = m->cfg.hidden;
    const uint32_t ffn = m->cfg.ffn;
    if (rows == 0 || rows > ffn) {
        rows = ffn;
    }
    const int t_gate = qrt_gemm_type_of(ffn, hidden, loc.gate);
    const int t_down = qrt_gemm_type_of(hidden, ffn, loc.down);
    if (t_gate < 0 || t_down < 0) {
        return fail(err, err_size, "layer %u expert %u has an unknown quant type", layer, expert);
    }
    out->gate_type = (uint32_t) t_gate;
    out->down_type = (uint32_t) t_down;

    unsigned char * blk = (unsigned char *) malloc((size_t) loc.bytes);
    unsigned char * a_qs = (unsigned char *) malloc((size_t) t * hidden);
    uint32_t * a_ds = (uint32_t *) malloc((size_t) t * (hidden / 32) * 4);
    unsigned char * h_qs = (unsigned char *) malloc((size_t) t * ffn);
    uint32_t * h_ds = (uint32_t *) malloc((size_t) t * (ffn / 32) * 4);
    float * gpu = (float *) malloc((size_t) t * hidden * 4);
    float * want = (float *) malloc((size_t) t * hidden * 4);
    if (blk == NULL || a_qs == NULL || a_ds == NULL || h_qs == NULL || h_ds == NULL ||
        gpu == NULL || want == NULL) {
        free(blk); free(a_qs); free(a_ds); free(h_qs); free(h_ds); free(gpu); free(want);
        return fail(err, err_size, "out of memory for the check buffers");
    }
    int rc = -1;
    VkBuffer tmp = VK_NULL_HANDLE;
    VkDeviceMemory tmp_mem = VK_NULL_HANDLE;
    qrt_vk_staging * stg = NULL;
    do {
        if (qrt_weights_read_expert(w, layer, expert, blk, (size_t) loc.bytes,
                                    err, err_size) != 0) {
            break;
        }
        if (qrt_vk_alloc(dev, loc.bytes, QRT_VK_USAGE_STORAGE | QRT_VK_USAGE_TRANSFER_DST,
                         &tmp, &tmp_mem, err, err_size) != 0) {
            break;
        }
        const uint64_t stg_bytes = (uint64_t) t * hidden * 4 > loc.bytes
                                   ? (uint64_t) t * hidden * 4 : loc.bytes;
        if (qrt_vk_staging_create(dev, stg_bytes, &stg, err, err_size) != 0) {
            break;
        }
        if (qrt_vk_upload(dev, stg, tmp, 0, blk, loc.bytes, err, err_size) != 0) {
            break;
        }
        fill_acts(t, hidden, a_qs, a_ds);

        // Gate: the matrix type varies by layer, the reference follows it.
        check_arm gate;
        memset(&gate, 0, sizeof(gate));
        gate.args.w = tmp;
        gate.args.wbase = (t_gate == QRT_QUANT_Q6K) ? 0u : 0u;
        gate.args.a_qs = m->a_qs;
        gate.args.a_ds = m->a_ds;
        gate.args.out = m->g_out;
        gate.args.n = ffn;
        gate.args.k = hidden;
        gate.args.m = t;
        gate.args.type = (qrt_quant) t_gate;
        gate.host = blk;
        gate.rows = rows > ffn ? ffn : rows;
        gate.type = (qrt_quant) t_gate;
        gate.gpu = gpu;
        gate.want = want;
        if (check_one(dev, m, &gate, a_qs, a_ds, &out->gate_rel_l1, err, err_size) != 0) {
            break;
        }

        // Down needs the swiglu output planes, so run gate, up and swiglu first and read
        // the hidden planes back, then check down against those exact planes.
        qrt_gemm_args up = gate.args;
        up.out = m->u_out;
        if (qrt_gemm_dispatch(m->gemm, &up, err, err_size) != 0 ||
            qrt_gemm_swiglu(m->gemm, m->g_out, m->u_out, m->h_qs, m->h_ds, t, ffn,
                            err, err_size) != 0 ||
            qrt_vk_download(dev, stg, m->h_qs, 0, h_qs, (uint64_t) t * ffn, err, err_size) != 0 ||
            qrt_vk_download(dev, stg, m->h_ds, 0, h_ds, (uint64_t) t * (ffn / 32) * 4,
                            err, err_size) != 0) {
            break;
        }
        check_arm down;
        memset(&down, 0, sizeof(down));
        down.args.w = tmp;
        down.args.wbase = (t_down == QRT_QUANT_Q6K) ? (uint32_t) loc.gate + (uint32_t) loc.up
                                                    : (uint32_t) (loc.gate + loc.up) / 4;
        down.args.a_qs = m->h_qs;
        down.args.a_ds = m->h_ds;
        down.args.out = m->o_out;
        down.args.n = hidden;
        down.args.k = ffn;
        down.args.m = t;
        down.args.type = (qrt_quant) t_down;
        down.host = blk + loc.gate + loc.up;
        down.rows = rows > hidden ? hidden : rows;
        down.type = (qrt_quant) t_down;
        down.gpu = gpu;
        down.want = want;
        if (check_one(dev, m, &down, h_qs, h_ds, &out->down_rel_l1, err, err_size) != 0) {
            break;
        }
        rc = 0;
    } while (0);
    qrt_vk_staging_destroy(dev, stg);
    if (tmp_mem != VK_NULL_HANDLE) {
        qrt_vk_free(dev, tmp, tmp_mem);
    }
    free(blk); free(a_qs); free(a_ds); free(h_qs); free(h_ds); free(gpu); free(want);
    return rc;
}


#else // QRT_HAVE_VULKAN

struct qrt_moe { int unused; };

int qrt_moe_open(qrt_vk_dev * dev, qrt_weights * w, const char * spv_dir,
                 const qrt_moe_cfg * cfg, qrt_moe ** out, char * err, size_t err_size) {
    (void) dev; (void) w; (void) spv_dir; (void) cfg;
    *out = NULL;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

void qrt_moe_close(qrt_vk_dev * dev, qrt_moe * m) { (void) dev; (void) m; }

uint32_t qrt_moe_tokens(const qrt_moe * m) { (void) m; return 0; }

uint64_t qrt_moe_flops(const qrt_moe * m) { (void) m; return 0; }

int qrt_moe_compute(void * user, uint32_t layer, uint32_t expert, VkBuffer buf,
                    uint64_t slot_offset, uint64_t bytes, char * err, size_t err_size) {
    (void) user; (void) layer; (void) expert; (void) buf; (void) slot_offset; (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_moe_check(qrt_vk_dev * dev, qrt_moe * m, qrt_weights * w, uint32_t layer,
                  uint32_t expert, uint32_t rows, qrt_moe_check_result * out, char * err,
                  size_t err_size) {
    (void) dev; (void) m; (void) w; (void) layer; (void) expert; (void) rows; (void) out;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

#endif // QRT_HAVE_VULKAN
