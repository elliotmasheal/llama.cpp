// M2-6c self check: the delta-rule recurrence against a double precision CPU
// reference, plus the two properties the task asks for explicitly: the state can
// be saved and restored, and a batched chunk equals a serial token-by-token run.
// q and k are random unit vectors because the prep chain L2 normalises them, so
// their direction is what matters; decay comes from the real ssm_a and ssm_dt.

#include "attn_deltanet.h"
#include "vk_transfer.h"

#include "gguf_index.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

struct rng_t {
    uint64_t s;
    uint64_t next(void) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double gauss(void) {
        const double u1 = ((double) ((next() >> 11) & ((1ull << 53) - 1)) + 0.5) / 9007199254740992.0;
        const double u2 = ((double) ((next() >> 11) & ((1ull << 53) - 1)) + 0.5) / 9007199254740992.0;
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    }
};

double rel_l1(const float * got, const float * want, size_t n, double * max_abs) {
    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double a = (double) got[i];
        const double b = (double) want[i];
        num += fabs(a - b);
        den += fabs(b);
        mx = fmax(mx, fabs(a - b));
    }
    *max_abs = mx;
    return den > 0.0 ? num / den : num;
}

} // namespace

int qrt_dn_check(const char * gguf_path, uint32_t layer, uint32_t tokens, const char * spv_dir) {
    char err[512] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open(0, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    qrt_gguf * g = NULL;
    if (qrt_gguf_open(gguf_path, &g, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }

    // Shapes from the tensor table: ssm_norm is head_dim, ssm_a is n_v_heads,
    // attn_qkv minus the v channels gives q and k (each n_k_heads * head_dim).
    qrt_dn_shapes sh;
    memset(&sh, 0, sizeof(sh));
    char name[128];
    std::vector<float> ssm_a, ssm_dt, ssm_norm;
    snprintf(name, sizeof(name), "blk.%u.ssm_a", layer);
    int shape_rc = qrt_gguf_read_f32(g, name, &ssm_a, err, sizeof(err));
    if (shape_rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", layer);
        shape_rc = qrt_gguf_read_f32(g, name, &ssm_dt, err, sizeof(err));
    }
    if (shape_rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_norm.weight", layer);
        shape_rc = qrt_gguf_read_f32(g, name, &ssm_norm, err, sizeof(err));
    }
    const qrt_gguf_tensor * qkv = NULL;
    const qrt_gguf_tensor * conv = NULL;
    if (shape_rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", layer);
        shape_rc = qrt_gguf_require(g, name, &qkv, err, sizeof(err));
    }
    if (shape_rc == 0) {
        snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", layer);
        shape_rc = qrt_gguf_require(g, name, &conv, err, sizeof(err));
    }
    if (shape_rc != 0) {
        fprintf(stderr, "qrt: %s\n", err[0] ? err : "layer tensors are missing");
        qrt_gguf_close(g);
        qrt_vk_close(dev);
        return 2;
    }
    sh.head_dim  = (uint32_t) ssm_norm.size();
    sh.n_v_heads = (uint32_t) ssm_a.size();
    sh.d_inner   = sh.n_v_heads * sh.head_dim;
    sh.n_embd    = (uint32_t) qkv->ne[0];
    sh.conv_ch   = (uint32_t) qkv->ne[1];
    sh.conv_taps = (uint32_t) conv->ne[0];
    sh.qk_ch     = (sh.conv_ch - sh.d_inner) / 2;
    sh.n_k_heads = sh.qk_ch / sh.head_dim;
    qrt_gguf_close(g);
    if (sh.n_k_heads == 0 || sh.qk_ch % sh.head_dim != 0 || sh.head_dim % 32 != 0) {
        fprintf(stderr, "qrt: layer %u does not split into q/k/v (qk_ch=%u head_dim=%u)\n",
                layer, sh.qk_ch, sh.head_dim);
        qrt_vk_close(dev);
        return 2;
    }
    printf("dn_device subgroup_size=%u subgroup_arithmetic=%d\n", dev->subgroup_size,
           dev->has_subgroup_arithmetic);

    const uint32_t max_tokens = tokens > 1024 ? tokens : 1024;
    const uint32_t serial_tokens = tokens < 64 ? tokens : 64;
    // the widest verified pass the design allows is k=8; the snapshot ring is sized for it
    const uint32_t snap_k = tokens < 8 ? tokens : 8;
    const uint64_t st_floats = (uint64_t) sh.n_v_heads * sh.head_dim * sh.head_dim;
    const uint64_t st_bytes = st_floats * 4;

    std::vector<float> q((size_t) max_tokens * sh.qk_ch), k((size_t) max_tokens * sh.qk_ch);
    std::vector<float> v((size_t) max_tokens * sh.d_inner);
    std::vector<float> decay((size_t) max_tokens * sh.n_v_heads);
    std::vector<float> beta((size_t) max_tokens * sh.n_v_heads);

    rng_t rng = { 0x51D3C7A19E4B2F65ull };
    for (uint32_t t = 0; t < max_tokens; ++t) {
        for (uint32_t h = 0; h < sh.n_k_heads; ++h) {
            float * qrow = q.data() + (size_t) t * sh.qk_ch + (size_t) h * sh.head_dim;
            float * krow = k.data() + (size_t) t * sh.qk_ch + (size_t) h * sh.head_dim;
            double qn = 0.0, kn = 0.0;
            for (uint32_t i = 0; i < sh.head_dim; ++i) {
                qrow[i] = (float) rng.gauss();
                krow[i] = (float) rng.gauss();
                qn += (double) qrow[i] * qrow[i];
                kn += (double) krow[i] * krow[i];
            }
            qn = sqrt(qn);
            kn = sqrt(kn);
            for (uint32_t i = 0; i < sh.head_dim; ++i) {
                qrow[i] = (float) (qrow[i] / qn);
                krow[i] = (float) (krow[i] / kn);
            }
        }
        for (uint32_t c = 0; c < sh.d_inner; ++c) {
            v[(size_t) t * sh.d_inner + c] = (float) rng.gauss();
        }
        for (uint32_t h = 0; h < sh.n_v_heads; ++h) {
            const double alpha = rng.gauss() + (double) ssm_dt[h];
            const double gain = (double) ssm_a[h] * log1p(exp(alpha));
            // ssm_a holds -exp(A_log), so exp(gain) is the per-token decay <= 1
            decay[(size_t) t * sh.n_v_heads + h] = (float) exp(gain);
            beta[(size_t) t * sh.n_v_heads + h] = (float) (1.0 / (1.0 + exp(-rng.gauss())));
        }
    }

    qrt_vk_staging * stg = NULL;
    // the widest upload is the prep chain's qkv plane (conv_ch per token), not y
    const uint64_t row_bytes = (uint64_t) (sh.conv_ch > sh.d_inner ? sh.conv_ch : sh.d_inner);
    const uint64_t stg_bytes = (uint64_t) max_tokens * row_bytes * 4 > st_bytes
                                 ? (uint64_t) max_tokens * row_bytes * 4 : st_bytes;
    if (qrt_vk_staging_create(dev, stg_bytes, &stg, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }

    int rc = 2;
    qrt_deltanet * dn = NULL;
    std::vector<float> zero_state((size_t) st_floats, 0.0f);
    std::vector<float> st_saved((size_t) st_floats), st_a((size_t) st_floats);
    std::vector<float> st_b((size_t) st_floats), st_serial((size_t) st_floats);
    std::vector<float> y_gpu((size_t) max_tokens * sh.d_inner);
    std::vector<float> y_cpu((size_t) max_tokens * sh.d_inner);
    std::vector<float> y_split((size_t) max_tokens * sh.d_inner);
    std::vector<double> st_cpu((size_t) st_floats);
    std::vector<float> st_cpu_f((size_t) st_floats);
    do {
        if (qrt_dn_open(dev, spv_dir, &sh, max_tokens, snap_k + 1u < 2u ? 2u : snap_k + 1u,
                        &dn, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            break;
        }
        const qrt_dn_bufs * b = qrt_dn_buffers(dn);

        // helpers: move a token range into the planes, then run and read back
        auto upload_range = [&](uint32_t t0, uint32_t count) -> int {
            if (qrt_vk_upload(dev, stg, b->q, 0, q.data() + (size_t) t0 * sh.qk_ch,
                              (uint64_t) count * sh.qk_ch * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->k, 0, k.data() + (size_t) t0 * sh.qk_ch,
                              (uint64_t) count * sh.qk_ch * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->v, 0, v.data() + (size_t) t0 * sh.d_inner,
                              (uint64_t) count * sh.d_inner * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->decay, 0, decay.data() + (size_t) t0 * sh.n_v_heads,
                              (uint64_t) count * sh.n_v_heads * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->beta, 0, beta.data() + (size_t) t0 * sh.n_v_heads,
                              (uint64_t) count * sh.n_v_heads * 4, err, sizeof(err)) != 0) {
                return -1;
            }
            return 0;
        };
        auto run_chunk = [&](uint32_t count, float * y_out, float * st_out) -> int {
            if (qrt_dn_recur(dn, count, err, sizeof(err)) != 0) {
                return -1;
            }
            if (qrt_vk_download(dev, stg, b->y, 0, y_out, (uint64_t) count * sh.d_inner * 4,
                                err, sizeof(err)) != 0) {
                return -1;
            }
            // the cursor already moved to the slot this chunk wrote
            return qrt_vk_download(dev, stg, b->state, qrt_dn_state_offset(dn), st_out, st_bytes,
                                   err, sizeof(err));
        };
        auto push_state = [&](const float * src) -> int {
            return qrt_vk_upload(dev, stg, b->state, qrt_dn_state_offset(dn), src, st_bytes, err,
                                 sizeof(err));
        };

        printf("dn_recur shapes n_embd=%u head_dim=%u k_heads=%u v_heads=%u d_inner=%u "
               "conv_ch=%u taps=%u tokens=%u\n", sh.n_embd, sh.head_dim, sh.n_k_heads,
               sh.n_v_heads, sh.d_inner, sh.conv_ch, sh.conv_taps, tokens);

        // 1-3) both recurrence variants: against the double precision reference,
        // state save and restore, and one token per dispatch
        std::fill(st_cpu.begin(), st_cpu.end(), 0.0);
        qrt_dn_recur_cpu(&sh, tokens, q.data(), k.data(), v.data(), decay.data(), beta.data(),
                         st_cpu.data(), y_cpu.data());
        for (size_t i = 0; i < st_cpu.size(); ++i) {
            st_cpu_f[i] = (float) st_cpu[i];
        }
        const uint32_t half = tokens / 2;
        const uint32_t rest = tokens - half;
        std::vector<float> y_row((size_t) sh.d_inner);
        std::vector<float> y_batch_prefix((size_t) serial_tokens * sh.d_inner);
        std::vector<float> st_batch_prefix((size_t) st_floats);
        int all_ok = 1;
        for (int variant = 0; variant < 2; ++variant) {
            const char * vname = variant ? "cols" : "serial";
            if (qrt_dn_set_recur(dn, variant) != 0) {
                printf("dn_recur variant %s unavailable on this device\n", vname);
                all_ok = 0;
                continue;
            }
            // 1) GPU against the double precision reference
            if (upload_range(0, tokens) != 0 || push_state(zero_state.data()) != 0 ||
                run_chunk(tokens, y_gpu.data(), st_a.data()) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                all_ok = 0;
                break;
            }
            double max_y = 0.0, max_s = 0.0;
            const double rel_y = rel_l1(y_gpu.data(), y_cpu.data(), (size_t) tokens * sh.d_inner, &max_y);
            const double rel_s = rel_l1(st_a.data(), st_cpu_f.data(), st_cpu_f.size(), &max_s);
            const int ok_ref = rel_y <= 1e-4 && rel_s <= 1e-4;
            printf("dn_recur correct variant=%s rel_l1=%.3e max_abs=%.3e state_rel_l1=%.3e "
                   "state_max_abs=%.3e verdict=%s\n", vname, rel_y, max_y, rel_s, max_s,
                   ok_ref ? "PASS" : "FAIL");
            all_ok &= ok_ref;

            // 2) state save and restore: two half chunks must equal one whole chunk
            if (push_state(zero_state.data()) != 0 || upload_range(0, half) != 0 ||
                run_chunk(half, y_split.data(), st_saved.data()) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                all_ok = 0;
                break;
            }
            // the saved state goes back in, the second half continues from it
            if (push_state(st_saved.data()) != 0 || upload_range(half, rest) != 0 ||
                run_chunk(rest, y_split.data(), st_b.data()) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                all_ok = 0;
                break;
            }
            double max_split = 0.0, max_state = 0.0;
            const double rel_split = rel_l1(y_split.data(), y_gpu.data() + (size_t) half * sh.d_inner,
                                            (size_t) rest * sh.d_inner, &max_split);
            const double rel_state = rel_l1(st_b.data(), st_a.data(), st_a.size(), &max_state);
            const int ok_split = rel_split == 0.0 && rel_state == 0.0;
            printf("dn_recur split   variant=%s half_tokens=%u rel_l1=%.3e max_abs=%.3e "
                   "state_rel_l1=%.3e verdict=%s\n", vname, half, rel_split, max_split, rel_state,
                   ok_split ? "PASS" : "FAIL");
            all_ok &= ok_split;

            // 3) serial decode: one token per dispatch must match the same chunk
            if (push_state(zero_state.data()) != 0 || upload_range(0, serial_tokens) != 0 ||
                run_chunk(serial_tokens, y_batch_prefix.data(), st_batch_prefix.data()) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                all_ok = 0;
                break;
            }
            int serial_fail = push_state(zero_state.data()) != 0;
            for (uint32_t t = 0; t < serial_tokens && !serial_fail; ++t) {
                // the ring cursor alternates, so the state stays where the next dispatch reads it
                if (upload_range(t, 1) != 0 || qrt_dn_recur(dn, 1, err, sizeof(err)) != 0 ||
                    qrt_vk_download(dev, stg, b->y, 0, y_row.data(),
                                    (uint64_t) sh.d_inner * 4, err, sizeof(err)) != 0 ||
                    qrt_vk_download(dev, stg, b->state, qrt_dn_state_offset(dn),
                                    st_serial.data(), st_bytes, err, sizeof(err)) != 0) {
                    serial_fail = 1;
                }
                if (!serial_fail) {
                    memcpy(y_split.data() + (size_t) t * sh.d_inner, y_row.data(),
                           (size_t) sh.d_inner * 4);
                }
            }
            if (serial_fail) {
                fprintf(stderr, "qrt: %s\n", err);
                all_ok = 0;
                break;
            }
            double max_serial = 0.0, max_serial_state = 0.0;
            const double rel_serial = rel_l1(y_split.data(), y_batch_prefix.data(),
                                             (size_t) serial_tokens * sh.d_inner, &max_serial);
            const double rel_serial_state = rel_l1(st_serial.data(), st_batch_prefix.data(),
                                                   st_batch_prefix.size(), &max_serial_state);
            const int ok_serial = rel_serial == 0.0 && rel_serial_state == 0.0;
            printf("dn_recur serial  variant=%s tokens=%u rel_l1=%.3e max_abs=%.3e state_rel_l1=%.3e "
                   "verdict=%s\n", vname, serial_tokens, rel_serial, max_serial, rel_serial_state,
                   ok_serial ? "PASS" : "FAIL");
            all_ok &= ok_serial;

            // 4) the snapshot ring (M3-3): one pass of snap_k rows must leave the state of
            //    every row, bit for bit the states the serial path writes, and reading back
            //    the snapshot of row j has to put the sequence exactly at row j.
            const uint32_t slots = qrt_dn_state_slots(dn);
            std::vector<float> st_snap((size_t) snap_k * st_floats);
            std::vector<float> st_ser((size_t) snap_k * st_floats);
            std::vector<float> y_snap((size_t) snap_k * sh.d_inner);
            std::vector<float> y_plain((size_t) snap_k * sh.d_inner);
            std::vector<float> st_plain((size_t) st_floats);
            std::vector<float> st_row((size_t) st_floats);
            std::vector<float> y_row1((size_t) sh.d_inner);
            std::vector<double> st_ref(st_floats, 0.0);
            std::vector<float> y_ref((size_t) snap_k * sh.d_inner);
            int snap_fail = 0;
            qrt_dn_state_use(dn, 0);
            if (snap_k + 1u > slots) {
                printf("dn_snap variant=%s skipped: the ring holds %u slots\n", vname, slots);
                all_ok = 0;
                continue;
            }
            if (push_state(zero_state.data()) != 0 || upload_range(0, snap_k) != 0 ||
                qrt_dn_recur_snap(dn, snap_k, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->y, 0, y_snap.data(),
                                (uint64_t) snap_k * sh.d_inner * 4, err, sizeof(err)) != 0) {
                snap_fail = 1;
            }
            for (uint32_t t = 0; !snap_fail && t < snap_k; ++t) {
                snap_fail = qrt_vk_download(dev, stg, b->state,
                                            (uint64_t) ((1u + t) % slots) * st_bytes,
                                            st_snap.data() + (size_t) t * st_floats, st_bytes,
                                            err, sizeof(err)) != 0;
            }
            // the same rows without snapshots: identical outputs, and the last snapshot is
            // the chunk's end state
            if (!snap_fail && (push_state(zero_state.data()) != 0 ||
                               run_chunk(snap_k, y_plain.data(), st_plain.data()) != 0)) {
                snap_fail = 1;
            }
            int ok_plain = !snap_fail &&
                           memcmp(y_plain.data(), y_snap.data(), y_plain.size() * 4) == 0 &&
                           memcmp(st_plain.data(), st_snap.data() + (size_t) (snap_k - 1u) * st_floats,
                                  st_plain.size() * 4) == 0;
            // the serial chain one token at a time, against every snapshot
            int ok_serial_snap = 0;
            if (!snap_fail) {
                qrt_dn_state_use(dn, 0);
                ok_serial_snap = push_state(zero_state.data()) == 0;
                for (uint32_t t = 0; ok_serial_snap && t < snap_k; ++t) {
                    ok_serial_snap = upload_range(t, 1) == 0 &&
                                     qrt_dn_recur(dn, 1, err, sizeof(err)) == 0 &&
                                     qrt_vk_download(dev, stg, b->state, qrt_dn_state_offset(dn),
                                                     st_ser.data() + (size_t) t * st_floats,
                                                     st_bytes, err, sizeof(err)) == 0 &&
                                     memcmp(st_ser.data() + (size_t) t * st_floats,
                                            st_snap.data() + (size_t) t * st_floats,
                                            (size_t) st_floats * 4) == 0;
                }
            }
            // and against the double precision reference, one token at a time
            double ref_max = 0.0, ref_rel = 0.0;
            int ok_snap_ref = 0;
            if (!snap_fail) {
                ok_snap_ref = 1;
                for (uint32_t t = 0; t < snap_k; ++t) {
                    qrt_dn_recur_cpu(&sh, 1, q.data() + (size_t) t * sh.qk_ch,
                                     k.data() + (size_t) t * sh.qk_ch,
                                     v.data() + (size_t) t * sh.d_inner,
                                     decay.data() + (size_t) t * sh.n_v_heads,
                                     beta.data() + (size_t) t * sh.n_v_heads, st_ref.data(),
                                     y_ref.data() + (size_t) t * sh.d_inner);
                    for (size_t i = 0; i < st_floats; ++i) {
                        st_row[i] = (float) st_ref[i];
                    }
                    double m = 0.0;
                    const double r = rel_l1(st_snap.data() + (size_t) t * st_floats,
                                            st_row.data(), st_floats, &m);
                    if (!(r <= 1e-4)) {
                        ok_snap_ref = 0;
                    }
                    if (m > ref_max) {
                        ref_max = m;
                    }
                    if (r > ref_rel) {
                        ref_rel = r;
                    }
                }
            }
            // rollback: put the cursor on the snapshot of row snap_k-2 and decode one more
            // token; it must reproduce that token's row of the snapshot pass bit for bit
            int ok_back = 0;
            if (!snap_fail && snap_k >= 2) {
                const uint32_t keep = snap_k - 2u;
                qrt_dn_state_use(dn, (1u + keep) % slots);
                ok_back = upload_range(keep + 1u, 1) == 0 &&
                          qrt_dn_recur(dn, 1, err, sizeof(err)) == 0 &&
                          qrt_vk_download(dev, stg, b->y, 0, y_row1.data(),
                                          (uint64_t) sh.d_inner * 4, err, sizeof(err)) == 0 &&
                          memcmp(y_row1.data(), y_snap.data() + (size_t) (keep + 1u) * sh.d_inner,
                                 (size_t) sh.d_inner * 4) == 0;
            }
            const int ok_snap = snap_fail == 0 && ok_plain && ok_serial_snap && ok_snap_ref && ok_back;
            printf("dn_snap variant=%s rows=%u plain_eq=%d serial_eq=%d ref_rel_l1=%.3e "
                   "ref_max=%.3e rollback_eq=%d verdict=%s\n", vname, snap_k, ok_plain,
                   ok_serial_snap, ref_rel, ref_max, ok_back, ok_snap ? "PASS" : "FAIL");
            if (snap_fail) {
                fprintf(stderr, "qrt: %s\n", err);
            }
            all_ok &= ok_snap;
        }
        if (all_ok == 0) {
            printf("dn_recur verdict FAIL (see the lines above)\n");
        }


        // 5) prep chain: conv1d(taps) + silu + per-head L2 norm against the CPU reference
        {
            qrt_dn_weights * w = NULL;
            const float l2_eps = 1e-6f / (float) sh.head_dim;   // attention.layer_norm_rms_epsilon / head_dim
            std::vector<float> qkv_host((size_t) tokens * sh.conv_ch);
            std::vector<float> cst((size_t) (sh.conv_taps - 1) * sh.conv_ch, 0.0f);
            for (uint32_t t = 0; t < tokens; ++t) {
                for (uint32_t c = 0; c < sh.conv_ch; ++c) {
                    qkv_host[(size_t) t * sh.conv_ch + c] = (float) rng.gauss();
                }
            }
            // a non-zero conv state exercises the history taps
            for (size_t i = 0; i < cst.size(); ++i) {
                cst[i] = (float) (rng.gauss() * 0.5);
            }
            std::vector<float> qc((size_t) tokens * sh.qk_ch), kc((size_t) tokens * sh.qk_ch);
            std::vector<float> vc((size_t) tokens * sh.d_inner);
            if (qrt_dn_weights_load(dev, gguf_path, layer, &sh, &w, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            std::vector<float> cw((size_t) sh.conv_taps * sh.conv_ch);
            if (qrt_vk_download(dev, stg, qrt_dn_weights_conv(w), 0, cw.data(), cw.size() * 4, err,
                                sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, w);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            // pattern probe: constant weights and inputs remove every data dependence, so a
            // wrong value here means the v path itself is broken, not the data
            {
                std::vector<float> ones_w((size_t) sh.conv_taps * sh.conv_ch, 1.0f);
                std::vector<float> ones_x((size_t) tokens * sh.conv_ch, 1.0f);
                std::vector<float> zero_s((size_t) (sh.conv_taps - 1) * sh.conv_ch, 0.0f);
                std::vector<float> qp((size_t) sh.qk_ch), kp((size_t) sh.qk_ch), vp((size_t) sh.d_inner);
                const int prc = qrt_vk_upload(dev, stg, qrt_dn_weights_conv(w), 0, ones_w.data(),
                                              ones_w.size() * 4, err, sizeof(err));
                const int prc2 = prc == 0 ? qrt_vk_upload(dev, stg, b->qkv, 0, ones_x.data(),
                                                         ones_x.size() * 4, err, sizeof(err)) : -1;
                const int prc3 = prc2 == 0 ? qrt_vk_upload(dev, stg, b->conv_state, 0,
                                                           zero_s.data(), zero_s.size() * 4, err,
                                                           sizeof(err)) : -1;
                if (prc3 == 0 && qrt_dn_prep_run(dn, w, tokens, l2_eps, err, sizeof(err)) == 0 &&
                    qrt_vk_download(dev, stg, b->q, 0, qp.data(), qp.size() * 4, err, sizeof(err)) == 0 &&
                    qrt_vk_download(dev, stg, b->v, 0, vp.data(), vp.size() * 4, err, sizeof(err)) == 0) {
                    // one tap of the conv is live (j = taps-1 reads this token), the rest read a
                    // zeroed state, so conv = 1 for every channel: silu(1) = 0.7310586
                    const double want_v = 1.0 / (1.0 + exp(-1.0));
                    const double want_q = want_v / (want_v * sqrt((double) sh.head_dim));
                    printf("dn_prep_pattern q0=%.6f (want %.6f) v0=%.6f v_mid=%.6f v_last=%.6f "
                           "(want %.6f) verdict=%s\n", (double) qp[0], want_q, (double) vp[0],
                           (double) vp[sh.d_inner / 2], (double) vp[sh.d_inner - 1], want_v,
                           (fabs((double) vp[0] - want_v) < 1e-4 &&
                            fabs((double) vp[sh.d_inner - 1] - want_v) < 1e-4 &&
                            fabs((double) qp[0] - want_q) < 1e-4) ? "PASS" : "FAIL");
                } else {
                    fprintf(stderr, "qrt: prep pattern probe failed: %s\n", err);
                }
                // put the real weights and inputs back
                if (qrt_vk_upload(dev, stg, qrt_dn_weights_conv(w), 0, cw.data(), cw.size() * 4,
                                  err, sizeof(err)) != 0 ||
                    qrt_vk_upload(dev, stg, b->qkv, 0, qkv_host.data(), qkv_host.size() * 4, err,
                                  sizeof(err)) != 0 ||
                    qrt_vk_upload(dev, stg, b->conv_state, 0, cst.data(), cst.size() * 4, err,
                                  sizeof(err)) != 0) {
                    qrt_dn_weights_free(dev, w);
                    fprintf(stderr, "qrt: %s\n", err);
                    break;
                }
            }

            std::vector<float> qg((size_t) tokens * sh.qk_ch), kg((size_t) tokens * sh.qk_ch);
            std::vector<float> vg((size_t) tokens * sh.d_inner);
            if (qrt_vk_upload(dev, stg, b->qkv, 0, qkv_host.data(),
                              qkv_host.size() * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->conv_state, 0, cst.data(), cst.size() * 4,
                              err, sizeof(err)) != 0 ||
                qrt_dn_prep_run(dn, w, tokens, l2_eps, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->q, 0, qg.data(), qg.size() * 4, err,
                                sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->k, 0, kg.data(), kg.size() * 4, err,
                                sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->v, 0, vg.data(), vg.size() * 4, err,
                                sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, w);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            double lo = 0.0, hi = 0.0;
            for (size_t i = 0; i < cw.size(); ++i) {
                (i < cw.size() / 2 ? lo : hi) += fabs((double) cw[i]);
            }
            qrt_dn_prep_cpu(&sh, tokens, qkv_host.data(), cw.data(), cst.data(), l2_eps,
                            qc.data(), kc.data(), vc.data(), NULL);
            double mq = 0.0, mk = 0.0, mv = 0.0;
            const double rq = rel_l1(qg.data(), qc.data(), qc.size(), &mq);
            const double rk = rel_l1(kg.data(), kc.data(), kc.size(), &mk);
            const double rv = rel_l1(vg.data(), vc.data(), vc.size(), &mv);
            const double prep_ms = qrt_dn_last_ms(dn);
            qrt_dn_weights_free(dev, w);
            printf("dn_prep  tokens=%u q_rel_l1=%.3e k_rel_l1=%.3e v_rel_l1=%.3e max_abs=%.3e "
                   "ms=%.3f verdict=%s\n", tokens, rq, rk, rv, fmax(mq, fmax(mk, mv)), prep_ms,
                   (rq <= 1e-4 && rk <= 1e-4 && rv <= 1e-4) ? "PASS" : "FAIL");
            if (!(rq <= 1e-4 && rk <= 1e-4 && rv <= 1e-4)) {
                break;
            }
        }

        // 5b) gate: decay = exp(a*softplus(alpha+dt)), beta = sigmoid, against the reference
        {
            qrt_dn_weights * gw = NULL;
            if (qrt_dn_weights_load(dev, gguf_path, layer, &sh, &gw, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            std::vector<float> xnorm((size_t) tokens * sh.n_embd);
            for (size_t i = 0; i < xnorm.size(); ++i) {
                xnorm[i] = (float) rng.gauss();
            }
            std::vector<float> dg((size_t) tokens * sh.n_v_heads), bg((size_t) tokens * sh.n_v_heads);
            std::vector<float> dc(dg.size()), bc(dg.size());
            if (qrt_vk_upload(dev, stg, b->norm, 0, xnorm.data(), xnorm.size() * 4, err,
                              sizeof(err)) != 0 ||
                qrt_dn_gate_run(dn, gw, tokens, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->decay, 0, dg.data(), dg.size() * 4, err,
                                sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->beta, 0, bg.data(), bg.size() * 4, err,
                                sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, gw);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            qrt_dn_gate_cpu(&sh, tokens, xnorm.data(), qrt_dn_weights_host(gw, 1), qrt_dn_weights_host(gw, 2),
                            qrt_dn_weights_host(gw, 3), qrt_dn_weights_host(gw, 4), dc.data(), bc.data());
            double md = 0.0, mb = 0.0;
            const double rd = rel_l1(dg.data(), dc.data(), dc.size(), &md);
            const double rb = rel_l1(bg.data(), bc.data(), bc.size(), &mb);
            const double gate_ms = qrt_dn_last_ms(dn);
            qrt_dn_weights_free(dev, gw);
            printf("dn_gate  tokens=%u decay_rel_l1=%.3e beta_rel_l1=%.3e max_abs=%.3e ms=%.3f "
                   "verdict=%s\n", tokens, rd, rb, fmax(md, mb), gate_ms,
                   (rd <= 1e-4 && rb <= 1e-4) ? "PASS" : "FAIL");
            if (!(rd <= 1e-4 && rb <= 1e-4)) {
                break;
            }
        }

        // 5c) gated RMS norm against the reference
        {
            qrt_dn_weights * gw = NULL;
            if (qrt_dn_weights_load(dev, gguf_path, layer, &sh, &gw, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            std::vector<float> yv((size_t) tokens * sh.d_inner), zv((size_t) sh.d_inner * tokens);
            for (size_t i = 0; i < yv.size(); ++i) {
                yv[i] = (float) rng.gauss();
            }
            for (size_t i = 0; i < zv.size(); ++i) {
                zv[i] = (float) rng.gauss();
            }
            std::vector<float> og((size_t) tokens * sh.d_inner, 0.0f), oc(og.size());
            if (qrt_vk_upload(dev, stg, b->y, 0, yv.data(), yv.size() * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->z, 0, zv.data(), zv.size() * 4, err, sizeof(err)) != 0 ||
                qrt_dn_gnorm_run(dn, gw, tokens, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->gnorm, 0, og.data(), og.size() * 4, err,
                                sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, gw);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            qrt_dn_gnorm_cpu(&sh, tokens, yv.data(), zv.data(), qrt_dn_weights_host(gw, 5), 1e-6f,
                             oc.data());
            double mo = 0.0;
            const double ro = rel_l1(og.data(), oc.data(), oc.size(), &mo);
            const double gn_ms = qrt_dn_last_ms(dn);
            qrt_dn_weights_free(dev, gw);
            printf("dn_gnorm tokens=%u rel_l1=%.3e max_abs=%.3e ms=%.3f verdict=%s\n", tokens, ro,
                   mo, gn_ms, ro <= 1e-4 ? "PASS" : "FAIL");
            if (ro > 1e-4) {
                break;
            }
        }

        // 5d) the four legs in sequence: one linear-attention layer as the model runs it
        {
            qrt_dn_weights * gw = NULL;
            if (qrt_dn_weights_load(dev, gguf_path, layer, &sh, &gw, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            // weights_load already read all six buffers back and compared them with the host
            printf("dn_weights layer=%u buffers=6 readback_mismatch=0\n", layer);
            const float l2_eps = 1e-6f / (float) sh.head_dim;
            std::vector<float> qkv((size_t) tokens * sh.conv_ch);
            for (size_t i = 0; i < qkv.size(); ++i) {
                qkv[i] = (float) rng.gauss();
            }
            std::vector<float> xnorm((size_t) tokens * sh.n_embd);
            for (size_t i = 0; i < xnorm.size(); ++i) {
                xnorm[i] = (float) rng.gauss();
            }
            std::vector<float> zv((size_t) sh.d_inner * tokens);
            for (size_t i = 0; i < zv.size(); ++i) {
                zv[i] = (float) rng.gauss();
            }
            std::vector<float> cst((size_t) (sh.conv_taps - 1) * sh.conv_ch, 0.0f);
            // no conv re-upload here: weights_load verifies every buffer against its host
            // copy, so a broken upload fails the load (DECISIONS 65)
            if (qrt_vk_upload(dev, stg, b->qkv, 0, qkv.data(), qkv.size() * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->conv_state, 0, cst.data(), cst.size() * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->norm, 0, xnorm.data(), xnorm.size() * 4, err, sizeof(err)) != 0 ||
                qrt_vk_upload(dev, stg, b->z, 0, zv.data(), zv.size() * 4, err, sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, gw);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            std::vector<float> zero_state_g((size_t) st_floats, 0.0f);
            if (qrt_vk_upload(dev, stg, b->state, qrt_dn_state_offset(dn), zero_state_g.data(),
                              st_bytes, err, sizeof(err)) != 0 ||
                qrt_dn_layer_run(dn, gw, tokens, l2_eps, 0, err, sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, gw);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            const double layer_ms = qrt_dn_layer_ms(dn);
            std::vector<float> og((size_t) tokens * sh.d_inner, 0.0f);
            std::vector<float> ychain((size_t) tokens * sh.d_inner, 0.0f);
            if (qrt_vk_download(dev, stg, b->gnorm, 0, og.data(), og.size() * 4, err, sizeof(err)) != 0 ||
                qrt_vk_download(dev, stg, b->y, 0, ychain.data(), ychain.size() * 4, err, sizeof(err)) != 0) {
                qrt_dn_weights_free(dev, gw);
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            // the reference is the same chain, one double precision piece after another
            std::vector<float> qr((size_t) tokens * sh.qk_ch), kr(qr.size()), vr((size_t) tokens * sh.d_inner);
            std::vector<float> csto((size_t) (sh.conv_taps - 1) * sh.conv_ch, 0.0f);
            std::vector<float> drows((size_t) tokens * sh.n_v_heads), brows(drows.size());
            std::vector<double> stc((size_t) st_floats, 0.0);
            std::vector<float> yref((size_t) tokens * sh.d_inner), oref((size_t) tokens * sh.d_inner);
            qrt_dn_prep_cpu(&sh, tokens, qkv.data(), qrt_dn_weights_host(gw, 0), cst.data(), l2_eps,
                            qr.data(), kr.data(), vr.data(), csto.data());
            qrt_dn_gate_cpu(&sh, tokens, xnorm.data(), qrt_dn_weights_host(gw, 1),
                            qrt_dn_weights_host(gw, 2), qrt_dn_weights_host(gw, 3),
                            qrt_dn_weights_host(gw, 4), drows.data(), brows.data());
            qrt_dn_recur_cpu(&sh, tokens, qr.data(), kr.data(), vr.data(), drows.data(), brows.data(),
                             stc.data(), yref.data());
            qrt_dn_gnorm_cpu(&sh, tokens, yref.data(), zv.data(), qrt_dn_weights_host(gw, 5), 1e-6f,
                             oref.data());
            double my = 0.0, mo = 0.0;
            const double ry = rel_l1(ychain.data(), yref.data(), yref.size(), &my);
            const double ro = rel_l1(og.data(), oref.data(), oref.size(), &mo);
            qrt_dn_weights_free(dev, gw);
            const double us_per_token = layer_ms * 1000.0 / (double) tokens;
            printf("dn_layer tokens=%u y_rel_l1=%.3e out_rel_l1=%.3e ms=%.3f us_per_token=%.2f "
                   "layers30_10k_ms=%.0f verdict=%s\n", tokens, ry, ro, layer_ms, us_per_token,
                   us_per_token * 10240.0 * 30.0 / 1000.0,
                   (ry <= 1e-4 && ro <= 1e-4) ? "PASS" : "FAIL");
            if (!(ry <= 1e-4 && ro <= 1e-4)) {
                break;
            }
        }

        // 4) timing at the chunk the prefill scheduler would use, for both variants
        double us_per_token[2] = { 0.0, 0.0 };
        int time_fail = 0;
        for (int v = 0; v < 2 && !time_fail; ++v) {
            qrt_dn_set_recur(dn, v);
            if (upload_range(0, max_tokens) != 0 || push_state(zero_state.data()) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                time_fail = 1;
                break;
            }
            double t[5];
            for (int i = 0; i < 5; ++i) {
                if (qrt_dn_recur(dn, max_tokens, err, sizeof(err)) != 0) {
                    fprintf(stderr, "qrt: %s\n", err);
                    time_fail = 1;
                    break;
                }
                t[i] = qrt_dn_last_ms(dn);
            }
            if (time_fail) {
                break;
            }
            for (int i = 0; i < 5; ++i) {
                for (int j = i + 1; j < 5; ++j) {
                    if (t[j] < t[i]) {
                        const double tmp = t[i];
                        t[i] = t[j];
                        t[j] = tmp;
                    }
                }
            }
            us_per_token[v] = t[2] * 1000.0 / (double) max_tokens;
            printf("dn_recur timing  variant=%s tokens=%u median_ms=%.3f us_per_token=%.2f "
                   "layer_10k_ms=%.1f layers30_10k_ms=%.0f\n",
                   v ? "cols" : "serial", max_tokens, t[2], us_per_token[v],
                   us_per_token[v] * 10240.0 / 1000.0, us_per_token[v] * 10240.0 * 30.0 / 1000.0);
        }
        if (time_fail) {
            break;
        }

        // 5) the two variants against the same double precision reference, and
        // against each other: they are different summation orders, so this is the
        // fp level difference a chunked scan would have to report under U-7.
        {
            std::vector<float> y_ser((size_t) tokens * sh.d_inner), y_col((size_t) tokens * sh.d_inner);
            std::vector<float> st_ser((size_t) st_floats), st_col((size_t) st_floats);
            int cmp_fail = upload_range(0, tokens) != 0;
            if (!cmp_fail) {
                qrt_dn_set_recur(dn, 0);
                cmp_fail = push_state(zero_state.data()) != 0 ||
                           qrt_dn_recur(dn, tokens, err, sizeof(err)) != 0 ||
                           qrt_vk_download(dev, stg, b->y, 0, y_ser.data(),
                                           (uint64_t) tokens * sh.d_inner * 4, err, sizeof(err)) != 0 ||
                           qrt_vk_download(dev, stg, b->state, qrt_dn_state_offset(dn),
                                           st_ser.data(), st_bytes, err, sizeof(err)) != 0;
            }
            if (!cmp_fail) {
                qrt_dn_set_recur(dn, 1);
                cmp_fail = push_state(zero_state.data()) != 0 ||
                           qrt_dn_recur(dn, tokens, err, sizeof(err)) != 0 ||
                           qrt_vk_download(dev, stg, b->y, 0, y_col.data(),
                                           (uint64_t) tokens * sh.d_inner * 4, err, sizeof(err)) != 0 ||
                           qrt_vk_download(dev, stg, b->state, qrt_dn_state_offset(dn),
                                           st_col.data(), st_bytes, err, sizeof(err)) != 0;
            }
            if (cmp_fail) {
                fprintf(stderr, "qrt: %s\n", err);
                break;
            }
            double m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;
            const double rs_cpu = rel_l1(y_ser.data(), y_cpu.data(), y_ser.size(), &m1);
            const double rc_cpu = rel_l1(y_col.data(), y_cpu.data(), y_col.size(), &m2);
            const double rc_ser = rel_l1(y_col.data(), y_ser.data(), y_col.size(), &m3);
            const double st_ser_col = rel_l1(st_col.data(), st_ser.data(), st_col.size(), &m4);
            printf("dn_recur compare serial_vs_ref=%.3e cols_vs_ref=%.3e cols_vs_serial=%.3e "
                   "state_cols_vs_serial=%.3e max_abs=%.3e verdict=%s\n",
                   rs_cpu, rc_cpu, rc_ser, st_ser_col, m3,
                   (rs_cpu <= 1e-4 && rc_cpu <= 1e-4) ? "PASS" : "FAIL");
            if (rs_cpu > 1e-4 || rc_cpu > 1e-4) {
                break;
            }
            qrt_dn_set_recur(dn, 0);
        }

        rc = all_ok ? 0 : 1;
    } while (0);

    qrt_dn_close(dev, dn);
    qrt_vk_staging_destroy(dev, stg);
    qrt_vk_close(dev);
    return rc;
}

#else // QRT_HAVE_VULKAN

int qrt_dn_check(const char * gguf_path, uint32_t layer, uint32_t tokens, const char * spv_dir) {
    (void) gguf_path;
    (void) layer;
    (void) tokens;
    (void) spv_dir;
    fprintf(stderr, "qrt: --deltanet-check needs a Vulkan build\n");
    return 2;
}

#endif // QRT_HAVE_VULKAN
