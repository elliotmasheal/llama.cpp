// Helpers shared by the K-quant GEMM kernels: the signed 4-lane dot, the packed
// 6-bit scale/min unpacking of Q4_K/Q5_K (get_scale_min_k4), and byte-addressed loads
// for Q6_K, whose 210-byte blocks are not a multiple of 4.

#ifndef QRT_GEMM_COMMON_GLSL
#define QRT_GEMM_COMMON_GLSL

#ifdef Q4K_DOT4
#define QRT_ENABLE_DOT
#endif

int qrt_dot4(uint q, uint a) {
#ifdef QRT_ENABLE_DOT
    return dotPacked4x8EXT(int(q), int(a));
#else
    const int p0 = int(q << 24u) >> 24;
    const int p1 = int(q << 16u) >> 24;
    const int p2 = int(q <<  8u) >> 24;
    const int p3 = int(q) >> 24;
    const int a0 = int(a << 24u) >> 24;
    const int a1 = int(a << 16u) >> 24;
    const int a2 = int(a <<  8u) >> 24;
    const int a3 = int(a) >> 24;
    return p0 * a0 + p1 * a1 + p2 * a2 + p3 * a3;
#endif
}

// Byte i of the 12 packed scale/min bytes, read as three words (Q4_K, Q5_K).
uint qrt_sm_byte(uvec3 s, uint i) {
    const uint v = (i < 4u) ? s.x : ((i < 8u) ? s.y : s.z);
    return (v >> ((i & 3u) * 8u)) & 0xFFu;
}

uint qrt_sm_scale(uvec3 s, uint j) {
    if (j < 4u) {
        return qrt_sm_byte(s, j) & 63u;
    }
    return (qrt_sm_byte(s, j + 4u) & 15u) | ((qrt_sm_byte(s, j - 4u) >> 6u) << 4);
}

uint qrt_sm_min(uvec3 s, uint j) {
    if (j < 4u) {
        return qrt_sm_byte(s, j + 4u) & 63u;
    }
    return (qrt_sm_byte(s, j + 4u) >> 4u) | ((qrt_sm_byte(s, j) >> 6u) << 4);
}

// fp32 -> binary16 with round-to-nearest-even. packHalf2x16 lowers to round-toward-zero on
// this driver, and any f16 a later kernel reads must round the way ggml's conversion does
// (DECISIONS 64.1).
uint qrt_f32_to_f16_rne(float f) {
    const uint x    = floatBitsToUint(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint mag  = x & 0x7FFFFFFFu;
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
        const uint shift = 126u - (mag >> 23);
        const uint mant  = (mag & 0x7FFFFFu) | 0x800000u;
        uint q = mant >> shift;
        const uint rem  = mant & ((1u << shift) - 1u);
        const uint hstep = 1u << (shift - 1u);
        if (rem > hstep || (rem == hstep && (q & 1u) != 0u)) {
            q++;
        }
        return sign | q;
    }
    uint mant = mag & 0x7FFFFFu;
    uint exp  = ((mag >> 23) - 127u) + 15u;
    const uint rem = mant & 0x1FFFu;
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

#endif
