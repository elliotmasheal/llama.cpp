// M0: dump per-layer top-k expert ids (ffn_moe_topk) for prompt prefill + greedy decode.
// Output: int32 header [n_tok, n_layer, k] then layer-major [layer][token][k].
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const int N_LAYER = 40;
static const int K_USED  = 8;

struct trace_state {
    std::vector<std::vector<int32_t>> ids;  // [layer] -> flat tokens*K
    std::vector<std::vector<int32_t>> last; // [layer] -> last argsort rows, for topk cross-check
    int n_layer = N_LAYER;
    long checked = 0;
    long mismatch = 0;
};

// ffn_moe_topk is a VIEW of ffn_moe_argsort in some decode graphs and is then never computed,
// so the trace reads the computed argsort node and keeps the first K_USED entries of each row.
static std::vector<int32_t> read_rows(struct ggml_tensor * t, int64_t keep) {
    const int64_t ne1 = t->ne[1];
    std::vector<char> host(ggml_nbytes(t));
    ggml_backend_tensor_get(t, host.data(), 0, host.size());
    std::vector<int32_t> buf(keep * ne1);
    for (int64_t j = 0; j < ne1; ++j) {
        for (int64_t k = 0; k < keep; ++k) {
            memcpy(&buf[j * keep + k], host.data() + j * t->nb[1] + k * t->nb[0], sizeof(int32_t));
        }
    }
    return buf;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (trace_state *) ud;
    const bool is_argsort = strncmp(t->name, "ffn_moe_argsort-", 16) == 0;
    const bool is_topk    = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (getenv("M0_DEBUG") && ask && strstr(t->name, "moe") && (strstr(t->name, "-3") || strstr(t->name, "-39"))) {
        fprintf(stderr, "DBG ask name=%s op=%s ne0=%lld ne1=%lld\n", t->name, ggml_op_name(t->op), (long long) t->ne[0], (long long) t->ne[1]);
    }
    if (!is_argsort && !is_topk) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (is_topk) {
        int tl = atoi(t->name + 13);
        if (tl < 0 || tl >= st->n_layer || t->type != GGML_TYPE_I32 || t->ne[0] != K_USED) {
            return false;
        }
        auto v = read_rows(t, K_USED);
        if (v.size() == st->last[tl].size()) {
            st->checked++;
            if (v != st->last[tl]) {
                st->mismatch++;
            }
        }
        return false;
    }
    int il = atoi(t->name + 16);
    if (il < 0 || il >= st->n_layer) {
        return false;
    }
    if (t->ne[0] < K_USED || t->type != GGML_TYPE_I32) {
        fprintf(stderr, "unexpected argsort tensor %s type=%d ne0=%lld\n", t->name, (int) t->type, (long long) t->ne[0]);
        return false;
    }
    auto buf = read_rows(t, K_USED);
    st->last[il] = buf;
    auto & dst = st->ids[il];
    dst.insert(dst.end(), buf.begin(), buf.end());
    return false;
}

static void decode_check(int rc, const char * what) {
    if (rc != 0) {
        fprintf(stderr, "%s failed rc=%d\n", what, rc);
        exit(1);
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: m0-trace <model.gguf> <prompt.txt> <n_gen> <out.bin>\n");
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt_path = argv[2];
    const int n_gen = atoi(argv[3]);
    const char * out_path = argv[4];

    std::ifstream pf(prompt_path, std::ios::binary);
    std::stringstream ss;
    ss << pf.rdbuf();
    std::string prompt = ss.str();

    llama_backend_init();

    trace_state st;
    st.ids.assign(N_LAYER, {});
    st.last.assign(N_LAYER, {});

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 12288;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "ctx failed\n");
        return 1;
    }

    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, false);
    std::vector<llama_token> toks(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), n_prompt, true, false);
    fprintf(stderr, "prompt tokens = %d\n", n_prompt);

    int n_total = 0;
    for (int i = 0; i < n_prompt; i += 512) {
        int n = std::min(512, n_prompt - i);
        decode_check(llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)), "prefill");
        n_total += n;
        fprintf(stderr, "prefill %d/%d\n", i + n, n_prompt);
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    int n_dec = 0;
    for (int g = 0; g < n_gen; ++g) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[best]) best = v;
        }
        if (llama_vocab_is_eog(vocab, best)) break;
        llama_token tk = best;
        decode_check(llama_decode(ctx, llama_batch_get_one(&tk, 1)), "decode");
        n_total += 1;
        n_dec += 1;
    }
    fprintf(stderr, "decoded %d tokens, total traced tokens %d\n", n_dec, n_total);
    fprintf(stderr, "topk cross-check: checked=%ld mismatch=%ld\n", st.checked, st.mismatch);

    // Per-layer record counts may be short (see M0 notes); the file stores each layer's own length.
    std::ofstream of(out_path, std::ios::binary);
    int32_t hdr[3] = { n_total, N_LAYER, K_USED };
    of.write((const char *) hdr, sizeof(hdr));
    for (int l = 0; l < N_LAYER; ++l) {
        const int32_t n_ints = (int32_t) st.ids[l].size();
        of.write((const char *) &n_ints, sizeof(n_ints));
        of.write((const char *) st.ids[l].data(), st.ids[l].size() * sizeof(int32_t));
        fprintf(stderr, "layer %2d tokens %d (expected %d)\n", l, n_ints / K_USED, n_total);
    }
    fprintf(stderr, "wrote %s\n", out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
