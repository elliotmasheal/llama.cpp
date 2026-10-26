#pragma once

#include <stddef.h>
#include <stdint.h>

// Small measurement helpers the check commands share, kept out of the CLI entry so a
// check that needs one does not drag in the whole command dispatch.

// Median of n samples; sorts v in place. Copy-call latency is submit+sync bound, so one
// sample is not a reading.
double qrt_check_median(double * v, int n);
void qrt_check_pattern_fill(unsigned char * p, size_t bytes, uint32_t salt);
int qrt_check_pattern_compare(const unsigned char * a, const unsigned char * b, size_t bytes);
