// Random expert block reads: the host side of the hybrid executor reads 8 experts of 1.883 MiB per
// layer, scattered over tens of GiB. A contiguous bandwidth number does not describe that access.
//
// build: g++ -O2 -std=c++17 -pthread bench/tools/moe_rand_bw.cpp -o bench/tools/moe_rand_bw.exe
// run:   moe_rand_bw.exe [buffer_MiB] [block_KiB] [blocks_per_run] [reps]
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    const size_t buf_mib   = argc > 1 ? (size_t) atoll(argv[1]) : 2048;
    const size_t block_kib = argc > 2 ? (size_t) atoll(argv[2]) : 1928;
    const size_t n_blocks  = argc > 3 ? (size_t) atoll(argv[3]) : 320;
    const int    reps      = argc > 4 ? atoi(argv[4]) : 5;

    const size_t buf_bytes   = buf_mib << 20;
    const size_t block_bytes = block_kib << 10;
    const size_t n_in_buf    = buf_bytes / block_bytes;

    std::vector<char> buf(buf_bytes);
    memset(buf.data(), 0x5a, buf_bytes); // fault in every page

    const int threads[] = {1, 2, 4, 5, 8};

    printf("buffer %zu MiB, block %zu KiB (%zu blocks in buffer), %zu blocks per run, reps %d\n",
           buf_mib, block_kib, n_in_buf, n_blocks, reps);

    for (int t : threads) {
        if ((size_t) t > n_blocks) {
            continue;
        }
        for (int mode = 0; mode < 2; ++mode) {
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                std::atomic<uint64_t> acc{0};
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<std::thread> ths;
                for (int i = 0; i < t; ++i) {
                    ths.emplace_back([&, i] {
                        std::mt19937_64 rng(0x9e3779b97f4a7c15ull * (i + 1) + r);
                        uint64_t sum = 0;
                        const size_t per = n_blocks / t;
                        // mode 0: scattered blocks; mode 1: the same volume read forward
                        size_t blk = mode == 0 ? (size_t) (rng() % n_in_buf) : (size_t) i * per;
                        for (size_t b = 0; b < per; ++b) {
                            const char * p = buf.data() + blk * block_bytes;
                            for (size_t off = 0; off < block_bytes; off += 64) {
                                sum += *(const uint64_t *) (p + off);
                            }
                            blk = mode == 0 ? (size_t) (rng() % n_in_buf) : (blk + 1) % n_in_buf;
                        }
                        acc.fetch_add(sum, std::memory_order_relaxed);
                    });
                }
                for (auto & th : ths) {
                    th.join();
                }
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (dt < best) {
                    best = dt;
                }
                if (acc.load() == 0xdeadbeefull) {
                    printf(" ");
                }
            }
            const double gbs = (double) n_blocks * block_bytes / best / 1e9;
            printf("%d threads %-8s %8.2f GB/s\n", t, mode == 0 ? "scatter" : "linear", gbs);
        }
    }
    return 0;
}
