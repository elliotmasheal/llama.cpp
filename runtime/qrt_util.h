#pragma once

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

#include <chrono>

// Low-level diagnostics shared by every runtime TU. Keeping them here stops each module
// from carrying its own copy with subtly different null handling.

// Formats into the caller's buffer and reports failure. A non-null buffer of non-zero
// size is the only thing written, matching what every call site already passes.
static inline int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (err != NULL && err_size != 0) {
        vsnprintf(err, err_size, fmt, ap);
    }
    va_end(ap);
    return -1;
}

// Milliseconds on a fixed monotonic epoch. Every caller takes differences, so the epoch
// itself is not part of any reading.
static inline double now_ms(void) {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}
