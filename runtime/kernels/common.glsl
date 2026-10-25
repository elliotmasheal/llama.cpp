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

#endif
