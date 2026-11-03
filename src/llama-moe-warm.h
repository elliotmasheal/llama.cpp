#pragma once

// Background page-cache warmer for host-resident expert weights.
//
// The MoE upload path reads missing experts straight from the mmap'd model
// file; cold pages turn every miss into a 4 KB random NVMe read (~3 GB/s
// effective). This thread keeps the expert pages resident in the system file
// cache (standby) so misses read from RAM instead. Opt-in via GGML_MOE_WARM=1;
// it is a pure page-cache hint with no numerical effect.

#include <atomic>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <memoryapi.h>
#include <thread>
#endif

struct llama_moe_warmup {
    std::thread th;
    std::atomic<bool> stop { false };

    void start(std::vector<std::pair<void *, size_t>> ranges) {
        th = std::thread([this, ranges = std::move(ranges)]() {
#if defined(_WIN32)
            HANDLE proc = GetCurrentProcess();
            std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
            entries.reserve(ranges.size());
            for (const auto & r : ranges) {
                entries.push_back({ r.first, r.second });
            }
            // yield disk and memory bandwidth to the inference thread
            SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
            while (!stop.load(std::memory_order_relaxed)) {
                MEMORYSTATUSEX ms {};
                ms.dwLength = sizeof(ms);
                if (GlobalMemoryStatusEx(&ms) && ms.ullAvailPhys > 10ull << 30) {
                    for (size_t i = 0; i < entries.size() && !stop.load(std::memory_order_relaxed); i += 32) {
                        const ULONG n = (ULONG) std::min<size_t>(32, entries.size() - i);
                        PrefetchVirtualMemory(proc, n, entries.data() + i, 0);
                        Sleep(50);
                    }
                }
                // refresh pass roughly every 120 s
                for (int k = 0; k < 240 && !stop.load(std::memory_order_relaxed); ++k) {
                    Sleep(500);
                }
            }
#endif
        });
    }

    ~llama_moe_warmup() {
        stop.store(true, std::memory_order_relaxed);
        if (th.joinable()) {
            th.join();
        }
    }
};
