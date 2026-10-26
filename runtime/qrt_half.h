#pragma once

#include <stdint.h>
#include <string.h>

// IEEE binary16 conversions shared by the kernels, the host copies and their references.
// The finite results are bit-identical to the ldexpf and malloc-free variants this
// runtime used before; NaN inputs stay NaN (the payload is not part of any gate).

static inline float f16_to_f32(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal: value = mant * 2^-24, so the exponent field is 103 + k with k the
            // top bit index; e counts the shifts so 113 - e is that field (RESULTS 72.4).
            int e = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                ++e;
            }
            bits = sign | ((uint32_t) (113 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Round-to-nearest-even fp32 -> binary16 bits. packHalf2x16 rounds toward zero on this
// driver (DECISIONS 64.1), so anything that has to match the device does this explicitly.
static inline uint32_t f32_to_f16_bits(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t mag  = x & 0x7FFFFFFFu;
    if (mag >= 0x7F800000u) {
        return sign | (mag > 0x7F800000u ? 0x7E00u : 0x7C00u);
    }
    if (mag >= 0x47800000u) {
        return sign | 0x7C00u;
    }
    if (mag < 0x38800000u) {
        if (mag < 0x33000000u) {
            return sign;
        }
        const uint32_t shift = 126u - (mag >> 23);
        const uint32_t mant  = (mag & 0x7FFFFFu) | 0x800000u;
        uint32_t q = mant >> shift;
        const uint32_t rem  = mant & ((1u << shift) - 1u);
        const uint32_t step = 1u << (shift - 1u);
        if (rem > step || (rem == step && (q & 1u) != 0u)) {
            q++;
        }
        return sign | q;
    }
    uint32_t mant = mag & 0x7FFFFFu;
    uint32_t exp  = ((mag >> 23) - 127u) + 15u;
    const uint32_t rem = mant & 0x1FFFu;
    mant >>= 13;
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u) != 0u)) {
        mant++;
        if (mant == 0x400u) {
            mant = 0u;
            exp++;
            if (exp >= 31u) {
                return sign | 0x7C00u;
            }
        }
    }
    return sign | (exp << 10u) | mant;
}
