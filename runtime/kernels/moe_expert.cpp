#include "kernels/moe_expert.h"

#include "kernels/q4k_gemm.h"
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
    qrt_q4k *     gemm;
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
    if (qrt_q4k_open(dev, spv_dir, &m->gemm, err, err_size) != 0) {
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
    qrt_q4k_close(dev, m->gemm);
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
    // gate and up only: the down matrix is Q5_K/Q6_K in this store.
    const uint64_t n = m->cfg.tokens;
    return 2ull * 2ull * n * m->cfg.ffn * m->cfg.hidden;
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

    qrt_q4k_gemm_args gate;
    memset(&gate, 0, sizeof(gate));
    gate.w = buf;
    gate.wbase_words = (uint32_t) (slot_offset / 4);
    gate.a_qs = m->a_qs;
    gate.a_ds = m->a_ds;
    gate.out = m->g_out;
    gate.n = ffn;
    gate.k = hidden;
    gate.m = t;

    qrt_q4k_gemm_args up = gate;
    up.wbase_words = (uint32_t) ((slot_offset + loc.gate) / 4);
    up.out = m->u_out;

    qrt_q4k_gemm_args down = gate;
    down.wbase_words = (uint32_t) ((slot_offset + loc.gate + loc.up) / 4);
    down.a_qs = m->h_qs;
    down.a_ds = m->h_ds;
    down.out = m->o_out;
    down.n = hidden;
    down.k = ffn;

    // The store mixes quant types: gate/up are Q4_K on most layers, down is Q5_K (or
    // Q6_K on layers 34/38/39, FORMAT.md). The GEMM here is Q4_K only, so a matrix of
    // another type is skipped and counted instead of read as garbage.
    const uint64_t q4k_gate = (uint64_t) ffn * (hidden / 256) * 144;
    const uint64_t q4k_down = (uint64_t) hidden * (ffn / 256) * 144;
    if (loc.gate != q4k_gate || loc.up != q4k_gate) {
        return fail(err, err_size, "expert %u,%u has gate/up of %llu/%llu bytes, expected %llu",
                    layer, expert, (unsigned long long) loc.gate,
                    (unsigned long long) loc.up, (unsigned long long) q4k_gate);
    }
    const int run_down = loc.down == q4k_down;
    if (!run_down) {
        ++m->down_skipped;
    }

    // One command buffer for all dispatches: at 64 tokens each GEMM is ~0.03 ms of math,
    // so four submit+fence rounds would cost more than the work (RESULTS 39).
    if (qrt_q4k_begin(m->gemm, err, err_size) != 0 ||
        qrt_q4k_record_gemm(m->gemm, &gate, err, err_size) != 0 ||
        qrt_q4k_record_gemm(m->gemm, &up, err, err_size) != 0 ||
        qrt_q4k_record_swiglu(m->gemm, m->g_out, m->u_out, m->h_qs, m->h_ds, t, ffn,
                              err, err_size) != 0 ||
        (run_down && qrt_q4k_record_gemm(m->gemm, &down, err, err_size) != 0)) {
        return -1;
    }
    if (qrt_q4k_submit(m->gemm, err, err_size) != 0) {
        return -1;
    }
    ++m->experts_run;
    return 0;
}

// Canonical Q4_K dequant (ggml dequantize_row_q4_K order) times the Q8_1 activation,
// in double, folded exactly like the kernel claims to fold it.
static void ref_gemm_row(const unsigned char * w, const unsigned char * a_qs,
                         const uint32_t * a_ds, uint32_t n_row, uint32_t k, uint32_t m_tok,
                         float * out) {
    const uint32_t nsb = k / 256;
    const uint32_t nk32 = k / 32;
    for (uint32_t tok = 0; tok < m_tok; ++tok) {
        double acc = 0.0;
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const unsigned char * blk = w + (size_t) n_row * nsb * 144 + sb * 144;
            const uint32_t dm = (uint32_t) blk[0] | ((uint32_t) blk[1] << 8);
            const uint32_t dmm = (uint32_t) blk[2] | ((uint32_t) blk[3] << 8);
            const float d = f16_to_f32(dm);
            const float dmin = f16_to_f32(dmm);
            const unsigned char * scales = blk + 4;
            const unsigned char * qs = blk + 16;
            for (uint32_t j = 0; j < 8; ++j) {
                uint32_t sc;
                uint32_t mn;
                if (j < 4) {
                    sc = scales[j] & 63u;
                    mn = scales[j + 4] & 63u;
                } else {
                    sc = (scales[j + 4] & 15u) | ((scales[j - 4] >> 6u) << 4);
                    mn = (scales[j + 4] >> 4u) | ((scales[j] >> 6u) << 4);
                }
                const uint32_t g = j / 2;
                const uint32_t half = j & 1u;
                const uint32_t abi = tok * nk32 + sb * 8 + j;
                const float aq = f16_to_f32(a_ds[abi] & 0xFFFFu);
                double sub = 0.0;
                for (uint32_t e = 0; e < 32; ++e) {
                    const uint8_t byte = qs[32 * g + e];
                    const uint32_t q = half == 0 ? (byte & 15u) : (byte >> 4u);
                    const int8_t qa = (int8_t) a_qs[(size_t) abi * 32 + e];
                    sub += ((double) d * (double) sc * (double) q - (double) dmin * (double) mn) *
                           ((double) aq * (double) qa);
                }
                acc += sub;
            }
        }
        out[tok] = (float) acc;
    }
}

int qrt_moe_check(qrt_vk_dev * dev, qrt_moe * m, qrt_weights * w, uint32_t layer,
                  uint32_t expert, uint32_t rows, double * rel_l1, char * err,
                  size_t err_size) {
    if (m == NULL || w == NULL || rel_l1 == NULL) {
        return fail(err, err_size, "the check needs the MoE leg and the store");
    }
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
    unsigned char * blk = (unsigned char *) malloc((size_t) loc.bytes);
    unsigned char * a_qs = (unsigned char *) malloc((size_t) t * hidden);
    uint32_t * a_ds = (uint32_t *) malloc((size_t) t * (hidden / 32) * 4);
    float * got = (float *) malloc((size_t) t * ffn * 4);
    float * want = (float *) malloc((size_t) t * rows * 4);
    if (blk == NULL || a_qs == NULL || a_ds == NULL || got == NULL || want == NULL) {
        free(blk); free(a_qs); free(a_ds); free(got); free(want);
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
        const uint64_t upload_bytes = loc.bytes > m->cfg.tokens * hidden + 4096
                                      ? loc.bytes : m->cfg.tokens * hidden + 4096;
        if (qrt_vk_staging_create(dev, upload_bytes, &stg, err, err_size) != 0) {
            break;
        }
        if (qrt_vk_upload(dev, stg, tmp, 0, blk, loc.bytes, err, err_size) != 0) {
            break;
        }
        qrt_q4k_gemm_args gate;
        memset(&gate, 0, sizeof(gate));
        gate.w = tmp;
        gate.wbase_words = 0;
        gate.a_qs = m->a_qs;
        gate.a_ds = m->a_ds;
        gate.out = m->g_out;
        gate.n = ffn;
        gate.k = hidden;
        gate.m = t;
        if (qrt_q4k_gemm(m->gemm, &gate, err, err_size) != 0) {
            break;
        }
        if (qrt_vk_download(dev, stg, m->g_out, 0, got, (uint64_t) t * ffn * 4,
                            err, err_size) != 0) {
            break;
        }
        // Same deterministic activations the open uploaded, regenerated on the host.
        fill_acts(t, hidden, a_qs, a_ds);
        for (uint32_t row = 0; row < rows; ++row) {
            float * row_out = want + (size_t) row * t;
            ref_gemm_row(blk, a_qs, a_ds, row, hidden, t, row_out);
        }
        double num = 0.0;
        double den = 0.0;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t tok = 0; tok < t; ++tok) {
                const double a = got[(size_t) row * t + tok];
                const double b = want[(size_t) row * t + tok];
                num += fabs(a - b);
                den += fabs(b);
            }
        }
        *rel_l1 = den > 0.0 ? num / den : num;
        rc = 0;
    } while (0);
    qrt_vk_staging_destroy(dev, stg);
    if (tmp_mem != VK_NULL_HANDLE) {
        qrt_vk_free(dev, tmp, tmp_mem);
    }
    free(blk); free(a_qs); free(a_ds); free(got); free(want);
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
                  uint32_t expert, uint32_t rows, double * rel_l1, char * err,
                  size_t err_size) {
    (void) dev; (void) m; (void) w; (void) layer; (void) expert; (void) rows; (void) rel_l1;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

#endif // QRT_HAVE_VULKAN
