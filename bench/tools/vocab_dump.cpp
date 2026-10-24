// M1-3a: tokenize a UTF-8 file, or emit greedy tokens, with the llama C API (CPU only, mmap).
// usage: vocab_dump MODEL tokens FILE [--add-special 0|1]
//        vocab_dump MODEL gen FILE N [--threads T] [--add-special 0|1]
// stdout: one token id per line. stderr: "#" diagnostics.
// Build (MinGW, static libs, see bench/tools/build_draft_vocab.py):
//   g++ -O2 -std=c++17 -fopenmp bench/tools/vocab_dump.cpp -Iinclude -Iggml/include
//   -Wl,--start-group build-vulkan/src/libllama.a build-vulkan/ggml/src/{ggml,ggml-cpu,ggml-base}.a
//   build-vulkan/ggml/src/ggml-vulkan/ggml-vulkan.a -Wl,--end-group -L<VULKAN_SDK>/Lib -lvulkan-1
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return true;
}

static std::vector<llama_token> tokenize(const struct llama_vocab * vocab, const std::string & text, bool add_special) {
    // Negative return is the required capacity; the +8 covers the empty/short-input edge cases.
    int32_t n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> toks((size_t) (n > 0 ? n : 1) + 8);
    n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), add_special, true);
    if (n < 0) {
        fprintf(stderr, "# tokenize failed: %d\n", n);
        exit(2);
    }
    toks.resize((size_t) n);
    return toks;
}

static void silent_log(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL tokens FILE | MODEL gen FILE N [--threads T] [--add-special 0|1]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string mode  = argv[2];
    const char * text_path  = argv[3];
    int n_gen       = mode == "gen" && argc > 4 ? atoi(argv[4]) : 0;
    int n_threads   = 4;
    int add_special = -1;
    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--add-special") == 0 && i + 1 < argc) {
            add_special = atoi(argv[++i]);
        }
    }

    std::string text;
    if (!read_file(text_path, text)) {
        fprintf(stderr, "# cannot read %s\n", text_path);
        return 2;
    }

    llama_log_set(silent_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP;
    mp.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "# model load failed: %s\n", model_path);
        return 3;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (add_special < 0) {
        add_special = llama_vocab_get_add_bos(vocab) ? 1 : 0;
    }
    fprintf(stderr, "# n_vocab=%d add_special=%d\n", llama_vocab_n_tokens(vocab), add_special);

    std::vector<llama_token> toks = tokenize(vocab, text, add_special != 0);
    fprintf(stderr, "# prompt_tokens=%zu\n", toks.size());

    if (mode == "tokens") {
        for (llama_token t : toks) {
            printf("%d\n", (int) t);
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }
    if (mode != "gen" || n_gen <= 0) {
        fprintf(stderr, "# unknown mode: %s\n", mode.c_str());
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) toks.size() + (uint32_t) n_gen + 8;
    cp.n_batch         = 2048;
    cp.n_ubatch        = 512;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf         = true;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "# context init failed\n");
        return 3;
    }

    const int32_t n_batch = 2048;
    const auto t_prefill0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < toks.size(); i += n_batch) {
        const int32_t n = (int32_t) std::min<size_t>(n_batch, toks.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) != 0) {
            fprintf(stderr, "# prefill decode failed at %zu\n", i);
            return 4;
        }
    }
    const auto t_prefill1 = std::chrono::steady_clock::now();

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    struct llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const auto t_gen0 = std::chrono::steady_clock::now();
    int produced = 0;
    bool hit_eog = false;
    for (int i = 0; i < n_gen; ++i) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, tok);
        printf("%d\n", (int) tok);
        ++produced;
        if (llama_vocab_is_eog(vocab, tok)) {
            hit_eog = true;
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
            fprintf(stderr, "# decode failed at generated %d\n", i);
            return 4;
        }
    }
    fprintf(stderr, "# generated=%d eog=%d\n", produced, hit_eog ? 1 : 0);
    const auto t_gen1 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(t_prefill1 - t_prefill0).count();
    const double gen_s     = std::chrono::duration<double>(t_gen1 - t_gen0).count();
    fprintf(stderr, "# prefill_s=%.2f prefill_tps=%.1f gen_s=%.2f gen_tps=%.2f\n",
            prefill_s, prefill_s > 0 ? (double) toks.size() / prefill_s : 0.0,
            gen_s, gen_s > 0 ? (double) produced / gen_s : 0.0);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
