#include "qrt_check_support.h"

#include <stdio.h>
#include <stdlib.h>

static int cmp_double(const void * a, const void * b) {
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return x < y ? -1 : (x > y ? 1 : 0);
}


// Median of n samples; the array is sorted in place. Copy-call latency on this
// GPU is dominated by submit+sync, so a single sample is not a reading.
double qrt_check_median(double * v, int n) {
    if (n <= 0) {
        return 0.0;
    }
    qsort(v, (size_t) n, sizeof(double), cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Fills with a position-dependent pattern so a partial copy or a stale staging
// byte shows up as a mismatch instead of matching zeros.
void qrt_check_pattern_fill(unsigned char * p, size_t bytes, uint32_t salt) {
    for (size_t i = 0; i < bytes; ++i) {
        p[i] = (unsigned char) ((i * 31u + salt + (i >> 8)) & 0xffu);
    }
}

int qrt_check_pattern_compare(const unsigned char * a, const unsigned char * b, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) {
        if (a[i] != b[i]) {
            printf("  first mismatch at byte %llu: %02x != %02x\n", (unsigned long long) i,
                   a[i], b[i]);
            return -1;
        }
    }
    return 0;
}
