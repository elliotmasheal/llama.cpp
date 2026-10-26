// CPU contract for the miss split policy (M3-2a).

#include "miss_split.h"

#include "qrt_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); \
} } while (0)

namespace {

// One layer with equal misses; returns its split and rolls the leg times into carry.
qrt_split_stats layer(const qrt_split_rates & rates, uint32_t count, uint64_t bytes,
                      uint32_t tokens, qrt_split_carry * carry,
                      std::vector<uint8_t> * mask) {
    std::vector<qrt_split_miss> misses(count);
    for (uint32_t i = 0; i < count; ++i) {
        misses[i].expert = i * 7u + 1u;
        misses[i].bytes = bytes;
        misses[i].tokens = tokens;
    }
    qrt_split_stats s{};
    char err[256] = {0};
    mask->assign(count, 0);
    CHECK(qrt_miss_split_plan(misses.data(), count, QRT_MISS_SPLIT_AUTO, &rates, carry,
                              mask->data(), &s, err, sizeof(err)) == 0);
    CHECK(s.h2d + s.cpu == count);
    CHECK(s.h2d_bytes == (uint64_t) s.h2d * bytes);
    carry->h2d_ms += s.h2d_ms;
    carry->cpu_ms += s.cpu_ms;
    return s;
}

void check_pass_balance(const qrt_split_carry & carry, double tolerance) {
    const double hi = carry.h2d_ms > carry.cpu_ms ? carry.h2d_ms : carry.cpu_ms;
    const double lo = carry.h2d_ms < carry.cpu_ms ? carry.h2d_ms : carry.cpu_ms;
    CHECK(hi > 0.0);
    CHECK((hi - lo) / hi <= tolerance);
}

} // namespace

int main() {
    qrt_split_rates rates;
    qrt_miss_split_default_rates(&rates);
    CHECK(rates.h2d_bytes_per_s > 0.0);
    CHECK(rates.cpu_ms[1] > 0.0);
    CHECK(rates.cpu_ms[5] > rates.cpu_ms[4]);
    CHECK(rates.cpu_ms[QRT_SPLIT_MAX_TOKENS] > rates.cpu_ms[5]);
    CHECK(qrt_miss_split_cpu_ms(&rates, 0) == rates.cpu_ms[1]);
    CHECK(qrt_miss_split_cpu_ms(&rates, QRT_SPLIT_MAX_TOKENS + 4) ==
          rates.cpu_ms[QRT_SPLIT_MAX_TOKENS]);

    char err[256] = {0};
    std::vector<uint8_t> mask;
    qrt_split_carry carry{0.0, 0.0};

    // A 40-layer pass of 8 equal misses per layer. One layer alone cannot split
    // within 10% (two granularity steps move a leg by about 20%), so the carry has
    // to repay that error on the next layers. The pass total must end balanced.
    uint32_t total_experts = 0;
    for (uint32_t l = 0; l < 40; ++l) {
        const qrt_split_stats s = layer(rates, 8, 1900544, 1, &carry, &mask);
        total_experts += s.h2d + s.cpu;
    }
    CHECK(total_experts == 320);
    check_pass_balance(carry, 0.10);
    CHECK(carry.h2d_ms > 0.0 && carry.cpu_ms > 0.0);

    // Same pass at t=5: the CPU expert is close to the H2D expert in time, so the
    // split is close to even and the carry still balances the pass.
    qrt_split_carry heavy_carry{0.0, 0.0};
    for (uint32_t l = 0; l < 40; ++l) {
        layer(rates, 8, 1900544, 5, &heavy_carry, &mask);
    }
    check_pass_balance(heavy_carry, 0.10);

    // One layer with 16 misses and no carry: still assigns both legs and preserves
    // the order and bytes accounting. Balance is not asserted for a single layer.
    qrt_split_carry single{0.0, 0.0};
    const qrt_split_stats s16 = layer(rates, 16, 1900544, 1, &single, &mask);
    CHECK(s16.h2d >= 1 && s16.cpu >= 1);
    CHECK(s16.h2d_bytes == (uint64_t) s16.h2d * 1900544u);
    uint32_t h2d_bits = 0;
    for (uint32_t i = 0; i < mask.size(); ++i) {
        h2d_bits += mask[i];
    }
    CHECK(h2d_bits == s16.h2d);

    // Determinism.
    qrt_split_carry again{0.0, 0.0};
    std::vector<uint8_t> mask2;
    const qrt_split_stats s16b = layer(rates, 16, 1900544, 1, &again, &mask2);
    CHECK(memcmp(&s16, &s16b, sizeof(s16)) == 0);
    CHECK(mask == mask2);

    // The carry is what moves the assignment: the CPU leg being behind sends this
    // layer's misses to the CPU.
    std::vector<qrt_split_miss> eight(8);
    for (uint32_t i = 0; i < eight.size(); ++i) {
        eight[i].expert = i;
        eight[i].bytes = 1900544;
        eight[i].tokens = 1;
    }
    std::vector<uint8_t> m_cpu(8, 0xFF), m_h2d(8, 0xFF);
    qrt_split_stats st{};
    qrt_split_carry cpu_behind{0.0, 10.0};
    CHECK(qrt_miss_split_plan(eight.data(), 8, QRT_MISS_SPLIT_AUTO, &rates, &cpu_behind,
                              m_cpu.data(), &st, err, sizeof(err)) == 0);
    CHECK(st.h2d == 8 && st.cpu == 0);
    qrt_split_carry h2d_behind{10.0, 0.0};
    CHECK(qrt_miss_split_plan(eight.data(), 8, QRT_MISS_SPLIT_AUTO, &rates, &h2d_behind,
                              m_h2d.data(), &st, err, sizeof(err)) == 0);
    CHECK(st.h2d == 0 && st.cpu == 8);

    // Fixed arms: 0 is all CPU, 100 is all H2D, 50 splits 8/8 on 16 misses.
    for (int pct : {0, 50, 100}) {
        qrt_split_stats fs{};
        std::vector<uint8_t> fm(16, 0xFF);
        CHECK(qrt_miss_split_plan(eight.data(), 8, pct, &rates, NULL, fm.data(), &fs, err,
                                  sizeof(err)) == 0);
        CHECK(fs.h2d + fs.cpu == 8);
        if (pct == 0) {
            CHECK(fs.h2d == 0 && fs.cpu == 8);
        } else if (pct == 100) {
            CHECK(fs.h2d == 8 && fs.cpu == 0);
        }
    }
    {
        qrt_split_stats fs{};
        std::vector<qrt_split_miss> sixteen(eight);
        for (uint32_t i = 0; i < 8; ++i) {
            qrt_split_miss m = eight[i];
            m.expert = i + 8;
            sixteen.push_back(m);
        }
        CHECK(qrt_miss_split_plan(sixteen.data(), 16, 50, &rates, NULL, NULL, &fs, err,
                                  sizeof(err)) == 0);
        CHECK(fs.h2d == 8 && fs.cpu == 8);
    }

    // Zero misses is valid, NULL list with a count is not.
    qrt_split_stats zero{};
    CHECK(qrt_miss_split_plan(NULL, 0, QRT_MISS_SPLIT_AUTO, &rates, NULL, NULL, &zero, err,
                              sizeof(err)) == 0);
    CHECK(zero.misses == 0 && zero.h2d == 0 && zero.cpu == 0);
    CHECK(qrt_miss_split_plan(NULL, 4, QRT_MISS_SPLIT_AUTO, &rates, NULL, NULL, &zero, err,
                              sizeof(err)) != 0);
    CHECK(qrt_miss_split_plan(eight.data(), 8, 101, &rates, NULL, NULL, &zero, err,
                              sizeof(err)) != 0);
    qrt_split_rates dead = rates;
    dead.h2d_bytes_per_s = 0.0;
    CHECK(qrt_miss_split_plan(eight.data(), 8, QRT_MISS_SPLIT_AUTO, &dead, NULL, NULL, &zero,
                              err, sizeof(err)) != 0);
    // The all-CPU control arm still works without an H2D rate.
    CHECK(qrt_miss_split_plan(eight.data(), 8, 0, &dead, NULL, NULL, &zero, err,
                              sizeof(err)) == 0);
    CHECK(zero.cpu == 8);

    // Rates follow observations without jumping to one sample.
    qrt_split_rates r2 = rates;
    qrt_miss_split_update(&r2, 4, 4ull * 1900544ull, 0.5, 3, 1, 0.6);
    CHECK(r2.h2d_bytes_per_s > rates.h2d_bytes_per_s);
    CHECK(r2.h2d_bytes_per_s < 2.0 * rates.h2d_bytes_per_s);
    const double cpu_before = r2.cpu_ms[1];
    qrt_miss_split_update(&r2, 0, 0, 0.0, 0, 0, 0.0);
    CHECK(r2.cpu_ms[1] == cpu_before);

    std::printf("miss_split_contract PASS cases=11\n");
    return 0;
}
