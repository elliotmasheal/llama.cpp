// Miss split policy (M3-2a). See miss_split.h for the contract.

#include "miss_split.h"

#include "qrt_config.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace {

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (err != NULL && err_size != 0) {
        vsnprintf(err, err_size, fmt, ap);
    }
    va_end(ap);
    return -1;
}

double h2d_ms(const qrt_split_rates * rates, uint64_t bytes) {
    return (double) bytes * 1000.0 / rates->h2d_bytes_per_s;
}

} // namespace

void qrt_miss_split_default_rates(qrt_split_rates * rates) {
    if (rates == NULL) {
        return;
    }
    memset(rates, 0, sizeof(*rates));
    // M1-7: H2D drops to 8.9-11.2 GB/s while the CPU leg reads RAM; 9.9 is the
    // middle of that measured band (design 2.6).
    rates->h2d_bytes_per_s = 9.9e9;
    // M3-1a (RESULTS 30/31), 8 threads: 0.079 / 0.106 / 0.160 / 0.199 ms at t=1/2/4/5.
    rates->cpu_ms[0] = 0.0;
    rates->cpu_ms[1] = 0.079;
    rates->cpu_ms[2] = 0.106;
    rates->cpu_ms[3] = 0.133;
    rates->cpu_ms[4] = 0.160;
    rates->cpu_ms[5] = 0.199;
    const double slope = rates->cpu_ms[5] - rates->cpu_ms[4];
    for (int t = 6; t <= QRT_SPLIT_MAX_TOKENS; ++t) {
        rates->cpu_ms[t] = rates->cpu_ms[t - 1] + slope;
    }
}

double qrt_miss_split_cpu_ms(const qrt_split_rates * rates, uint32_t tokens) {
    if (rates == NULL) {
        return 0.0;
    }
    if (tokens == 0) {
        tokens = 1;
    }
    if (tokens > QRT_SPLIT_MAX_TOKENS) {
        tokens = QRT_SPLIT_MAX_TOKENS;
    }
    return rates->cpu_ms[tokens];
}

int qrt_miss_split_plan(const qrt_split_miss * misses, uint32_t count, int percent,
                        const qrt_split_rates * rates, const qrt_split_carry * carry,
                        uint8_t * h2d_mask, qrt_split_stats * out_stats,
                        char * err, size_t err_size) {
    if (out_stats == NULL || rates == NULL) {
        return fail(err, err_size, "miss split needs rates and a statistics row");
    }
    if (count != 0 && misses == NULL) {
        return fail(err, err_size, "miss split needs the miss list");
    }
    const int auto_split = percent == QRT_MISS_SPLIT_AUTO;
    if (!auto_split && (percent < 0 || percent > 100)) {
        return fail(err, err_size, "miss split %d is neither auto nor 0..100", percent);
    }
    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->misses = count;
    if (count == 0) {
        return 0;
    }
    const double cpu_ok = rates->cpu_ms[1] > 0.0;
    if (!auto_split && percent > 0 && rates->h2d_bytes_per_s <= 0.0) {
        return fail(err, err_size, "miss split needs a positive H2D rate");
    }
    if (!auto_split && percent < 100 && !cpu_ok) {
        return fail(err, err_size, "miss split needs a positive CPU rate");
    }
    if (auto_split && (rates->h2d_bytes_per_s <= 0.0 || !cpu_ok)) {
        return fail(err, err_size, "auto miss split needs both leg rates");
    }

    std::vector<uint8_t> mask(count, 0);
    if (auto_split) {
        // Greedy LPT: handle the largest miss first and put it on the leg that the
        // addition hurts least. With one layer's handful of misses this keeps the
        // two legs within a few percent; experts in a layer share their byte size,
        // so the multiplicity is what makes the order non-uniform.
        struct item { uint32_t index; double h; double c; };
        std::vector<item> order(count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tokens = misses[i].tokens == 0 ? 1 : misses[i].tokens;
            order[i].index = i;
            order[i].h = h2d_ms(rates, misses[i].bytes);
            order[i].c = qrt_miss_split_cpu_ms(rates, tokens);
        }
        std::sort(order.begin(), order.end(), [](const item & a, const item & b) {
            const double am = a.h > a.c ? a.h : a.c;
            const double bm = b.h > b.c ? b.h : b.c;
            if (am != bm) {
                return am > bm;
            }
            return a.index < b.index;
        });
        double h_load = carry != NULL ? carry->h2d_ms : 0.0;
        double c_load = carry != NULL ? carry->cpu_ms : 0.0;
        for (size_t k = 0; k < order.size(); ++k) {
            const item & it = order[k];
            // Prefer H2D on ties: it leaves the CPU leg free for the next layer.
            if (h_load <= c_load) {
                mask[it.index] = 1;
                h_load += it.h;
            } else {
                c_load += it.c;
            }
        }
    } else {
        uint32_t n_h2d = (uint32_t) (((uint64_t) count * (uint32_t) percent + 50u) / 100u);
        for (uint32_t i = 0; i < n_h2d && i < count; ++i) {
            mask[i] = 1;
        }
    }

    for (uint32_t i = 0; i < count; ++i) {
        if (mask[i]) {
            ++out_stats->h2d;
            out_stats->h2d_bytes += misses[i].bytes;
            out_stats->h2d_ms += h2d_ms(rates, misses[i].bytes);
        } else {
            ++out_stats->cpu;
            out_stats->cpu_ms += qrt_miss_split_cpu_ms(rates, misses[i].tokens);
        }
    }
    if (h2d_mask != NULL) {
        memcpy(h2d_mask, mask.data(), count);
    }
    return 0;
}

void qrt_miss_split_update(qrt_split_rates * rates, uint32_t h2d_experts,
                           uint64_t h2d_bytes, double h2d_ms_obs, uint32_t cpu_experts,
                           uint32_t cpu_tokens, double cpu_ms_obs) {
    if (rates == NULL) {
        return;
    }
    if (h2d_experts != 0 && h2d_bytes != 0 && h2d_ms_obs > 0.0) {
        const double observed = (double) h2d_bytes * 1000.0 / h2d_ms_obs;
        rates->h2d_bytes_per_s = 0.75 * rates->h2d_bytes_per_s + 0.25 * observed;
    }
    if (cpu_experts != 0 && cpu_ms_obs > 0.0) {
        if (cpu_tokens == 0) {
            cpu_tokens = 1;
        }
        if (cpu_tokens > QRT_SPLIT_MAX_TOKENS) {
            cpu_tokens = QRT_SPLIT_MAX_TOKENS;
        }
        const double observed = cpu_ms_obs / (double) cpu_experts;
        rates->cpu_ms[cpu_tokens] = 0.75 * rates->cpu_ms[cpu_tokens] + 0.25 * observed;
    }
}
