#pragma once

// Lightweight, opt-in performance tracing for llama/ggml.
//
// Enable with GGML_PERF_TRACE=1 (aggregate summary on stderr at exit) or =2
// (aggregate + per-event CSV trace). CSV file defaults to perf-trace.csv in
// the CWD, override with GGML_PERF_TRACE_FILE.
//
// All events are no-ops (single relaxed atomic load) when disabled, so the
// instrumented build behaves identically to the baseline.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ggml_perf {

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int level() {
    static const int lvl = [] {
        const char * e = getenv("GGML_PERF_TRACE");
        return e ? atoi(e) : 0;
    }();
    return lvl;
}

inline bool enabled() { return level() > 0; }

struct entry {
    uint64_t count    = 0;
    uint64_t total_ns = 0;
    uint64_t bytes    = 0; // bytes moved (uploads/downloads/copies)
    uint64_t aux      = 0; // event-specific sub-count (experts, nodes, ...)
};

// key = user-defined discriminator (layer index, split id, ...), shown in the trace
inline void event(const char * cat, const char * name, int64_t key, uint64_t dur_ns, uint64_t bytes = 0, uint64_t aux = 0);

// RAII timer
struct scope {
    const char * cat;
    const char * name;
    int64_t      key;
    uint64_t     bytes;
    uint64_t     aux;
    int64_t      t0;

    scope(const char * c, const char * n, int64_t k = 0, uint64_t b = 0, uint64_t a = 0) :
            cat(c), name(n), key(k), bytes(b), aux(a), t0(now_ns()) {}

    ~scope() {
        if (enabled()) {
            event(cat, name, key, (uint64_t)(now_ns() - t0), bytes, aux);
        }
    }
};

namespace impl {

struct table {
    std::map<std::pair<std::string, std::string>, entry> map;
    std::mutex   mtx;
    std::FILE  * trace = nullptr;
    int64_t      base  = 0;

    static table & get() {
        static table t;
        return t;
    }

    void trace_line(const char * cat, const char * name, int64_t key, uint64_t dur_ns, uint64_t bytes, uint64_t aux) {
        const int64_t now = now_ns();
        if (trace == nullptr) {
            const char * fn = getenv("GGML_PERF_TRACE_FILE");
            trace = std::fopen(fn ? fn : "perf-trace.csv", "w");
            base  = now;
            if (trace) {
                std::fprintf(trace, "t_us,cat,name,key,dur_us,bytes,aux\n");
            }
        }
        if (trace) {
            std::fprintf(trace, "%.1f,%s,%s,%lld,%.2f,%zu,%zu\n",
                (now - base)/1e3, cat, name, (long long)key, dur_ns/1e3, (size_t)bytes, (size_t)aux);
        }
    }
};

inline void dump() {
    table & t = table::get();
    std::lock_guard<std::mutex> lock(t.mtx);
    if (t.map.empty()) {
        return;
    }
    std::fprintf(stderr, "\n== ggml perf trace (GGML_PERF_TRACE) ==\n");
    std::fprintf(stderr, "%-10s %-22s %10s %12s %10s %14s %10s\n",
        "cat", "name", "count", "total_ms", "avg_us", "bytes", "aux");
    for (const auto & kv : t.map) {
        const entry & e = kv.second;
        std::fprintf(stderr, "%-10s %-22s %10llu %12.3f %10.2f %14llu %10llu\n",
            kv.first.first.c_str(), kv.first.second.c_str(),
            (unsigned long long)e.count, e.total_ns/1e6, e.total_ns/1e3/e.count,
            (unsigned long long)e.bytes, (unsigned long long)e.aux);
    }
    if (t.trace) {
        std::fclose(t.trace);
        t.trace = nullptr;
        std::fprintf(stderr, "trace csv written\n");
    }
    std::fflush(stderr);
}

} // namespace impl

inline void event(const char * cat, const char * name, int64_t key, uint64_t dur_ns, uint64_t bytes, uint64_t aux) {
    if (!enabled()) {
        return;
    }
    static const bool reg = [] { std::atexit(impl::dump); return true; }();
    (void) reg;
    impl::table & t = impl::table::get();
    std::lock_guard<std::mutex> lock(t.mtx);
    entry & e = t.map[{cat, name}];
    e.count    += 1;
    e.total_ns += dur_ns;
    e.bytes    += bytes;
    e.aux      += aux;
    if (level() >= 2) {
        t.trace_line(cat, name, key, dur_ns, bytes, aux);
    }
}

// counter-only event (no duration)
inline void count(const char * cat, const char * name, uint64_t aux = 0) {
    event(cat, name, 0, 0, 0, aux);
}

} // namespace ggml_perf
