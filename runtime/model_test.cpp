// Build with qrt_config.cpp, without qrt_core or QRT_HAVE_VULKAN.
#ifdef QRT_HAVE_VULKAN
#error model_test must not link the Vulkan backend
#endif

#include "model.h"
#include "model_file.h"
#include "model_io.h"
#include "attn_full.h"
#include "ffn.h"
#include "gguf_index.h"
#include "layer.h"
#include "qrt_config.h"
#include "vk_dev.h"
#include "vk_transfer.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace model_test {

void require(bool ok, const char * text) {
    if (!ok) {
        fprintf(stderr, "model_test failed: %s\n", text);
        abort();
    }
}

struct state {
    std::string failure;
    unsigned failure_call = 1;
    bool partial_create = false;
    std::map<std::string, unsigned> calls;
    std::vector<std::string> owners;
    unsigned acquired = 0;
    unsigned released = 0;
    unsigned file_live = 0;
    unsigned file_opened = 0;
    unsigned file_closed = 0;
    unsigned short_write_at = 0;
    unsigned writes = 0;
    unsigned get_count = 0;
    unsigned read_error_at = 0;
    bool close_error = false;
    bool open_error = false;
    bool omit_head = false;
    float ref_bias = 0.0f;
    qrt_gguf_hparams hp{};
    std::map<std::string, std::vector<unsigned char>> files;
    std::vector<uint32_t> tiers;
    std::vector<qrt_layer_cfg> layer_configs;
    std::vector<uint32_t> layer_chunks;
    std::vector<uint32_t> positions;
    std::vector<uint32_t> lengths;
    std::vector<std::vector<float>> layer_inputs;
    std::vector<std::vector<float>> ref_inputs;
    std::vector<int> ref_planes;
};

state s;

int hit(const char * name, char * err, size_t cap) {
    const unsigned n = ++s.calls[name];
    if (s.failure == name && n == s.failure_call) {
        snprintf(err, cap, "injected %s", name);
        return -1;
    }
    return 0;
}

void acquire(const char * name) {
    s.owners.push_back(name);
    ++s.acquired;
}

void release(const char * name) {
    require(!s.owners.empty() && s.owners.back() == name, "resource destruction order");
    s.owners.pop_back();
    ++s.released;
}

void balanced() {
    require(s.owners.empty() && s.acquired == s.released, "all resources released");
    require(s.file_live == 0 && s.file_opened == s.file_closed, "all files closed");
}

void reset() {
    balanced();
    s = state{};
    s.hp.n_embd = 32;
    s.hp.block_count = 3;
    s.hp.nextn = 1;
    s.hp.present = QRT_HP_N_EMBD | QRT_HP_BLOCK_COUNT | QRT_HP_NEXTN;
}

} // namespace model_test

struct qrt_model_file {
    std::string path;
    std::vector<unsigned char> bytes;
    size_t pos = 0;
    bool reading = false;
    bool read_error = false;
};

// The model module's file port, replaced here so no production path is written and every
// short write, read error and failed close is an injected event.
qrt_model_file * qrt_model_file_open(const char * path, const char * mode) {
    if (model_test::s.open_error) return nullptr;
    const bool reading = mode[0] == 'r';
    const auto it = model_test::s.files.find(path);
    if (reading && it == model_test::s.files.end()) return nullptr;
    qrt_model_file * f = new qrt_model_file;
    f->path = path;
    f->reading = reading;
    if (reading) f->bytes = it->second;
    ++model_test::s.file_live;
    ++model_test::s.file_opened;
    return f;
}

int qrt_model_file_close(qrt_model_file * f) {
    if (f == nullptr) return EOF;
    if (!f->reading) model_test::s.files[f->path] = f->bytes;
    delete f;
    --model_test::s.file_live;
    ++model_test::s.file_closed;
    return model_test::s.close_error ? EOF : 0;
}

size_t qrt_model_file_write(qrt_model_file * f, const void * src, size_t size, size_t count) {
    model_test::require(!f->reading && count > 0 && size > 0, "valid write request");
    ++model_test::s.writes;
    const size_t written = model_test::s.writes == model_test::s.short_write_at ? count - 1 : count;
    const unsigned char * p = static_cast<const unsigned char *>(src);
    f->bytes.insert(f->bytes.end(), p, p + written * size);
    return written;
}

int qrt_model_file_getc(qrt_model_file * f) {
    model_test::require(f->reading, "read from input stream");
    if (++model_test::s.get_count == model_test::s.read_error_at) {
        f->read_error = true;
        return EOF;
    }
    return f->pos < f->bytes.size() ? f->bytes[f->pos++] : EOF;
}

int qrt_model_file_error(qrt_model_file * f) {
    return f->read_error ? 1 : 0;
}

struct VkBuffer_T {
    std::vector<float> values;
};

struct qrt_vk_staging {
    uint64_t bytes;
};

struct qrt_gguf {
    qrt_gguf_hparams hp;
    qrt_gguf_tensor tensors[3];
};

struct qrt_layer {
    uint32_t index = 0;
    uint32_t elapsed = 0;
    qrt_layer_info info{};
    std::vector<float> input;
    VkBuffer_T output;
    bool captured = false;
};

int qrt_vk_open(uint32_t tier, qrt_vk_dev ** out, char * err, size_t cap) {
    *out = nullptr;
    model_test::s.tiers.push_back(tier);
    const int rc = model_test::hit("dev_open", err, cap);
    if (rc == 0 || model_test::s.partial_create) {
        *out = new qrt_vk_dev{};
        (*out)->tier_mib = tier;
        model_test::acquire("dev");
    }
    return rc;
}

void qrt_vk_close(qrt_vk_dev * dev) {
    model_test::require(dev != nullptr, "close valid device");
    model_test::release("dev");
    delete dev;
}

int qrt_vk_staging_create(qrt_vk_dev *, uint64_t bytes, qrt_vk_staging ** out,
                          char * err, size_t cap) {
    *out = nullptr;
    const int rc = model_test::hit("staging_create", err, cap);
    if (rc == 0 || model_test::s.partial_create) {
        *out = new qrt_vk_staging{bytes};
        model_test::acquire("staging");
    }
    return rc;
}

void qrt_vk_staging_destroy(qrt_vk_dev * dev, qrt_vk_staging * stg) {
    model_test::require(dev != nullptr && stg != nullptr, "destroy staging with live device");
    model_test::release("staging");
    delete stg;
}

int qrt_vk_download(qrt_vk_dev *, qrt_vk_staging * stg, VkBuffer src, uint64_t offset,
                    void * dst, uint64_t bytes, char * err, size_t cap) {
    if (model_test::hit("download", err, cap) != 0) return -1;
    model_test::require(src != nullptr && bytes <= stg->bytes, "download fits staging");
    model_test::require(offset + bytes <= src->values.size() * sizeof(float), "download fits source");
    memcpy(dst, reinterpret_cast<const unsigned char *>(src->values.data()) + offset, (size_t) bytes);
    return 0;
}

int qrt_gguf_open(const char *, qrt_gguf ** out, char * err, size_t cap) {
    *out = nullptr;
    const int rc = model_test::hit("gguf_open", err, cap);
    if (rc != 0 && !model_test::s.partial_create) return rc;
    qrt_gguf * g = new qrt_gguf{};
    g->hp = model_test::s.hp;
    const char * names[] = {"token_embd.weight", "output.weight", "output_norm.weight"};
    for (unsigned i = 0; i < 3; ++i) {
        qrt_gguf_tensor & t = g->tensors[i];
        t.name = const_cast<char *>(names[i]);
        t.type = i == 2 ? 0 : 8;
        t.n_dims = i == 2 ? 1 : 2;
        t.ne[0] = 32;
        t.ne[1] = i == 2 ? 1 : 3;
        t.offset = 1024u + 1024u * i;
        t.nbytes = i == 2 ? 128 : 102;
    }
    *out = g;
    model_test::acquire("gguf");
    return rc;
}

void qrt_gguf_close(qrt_gguf * g) {
    model_test::require(g != nullptr, "close valid GGUF");
    model_test::release("gguf");
    delete g;
}

const qrt_gguf_hparams * qrt_gguf_hparams_of(const qrt_gguf * g) { return &g->hp; }

const qrt_gguf_tensor * qrt_gguf_find(const qrt_gguf * g, const char * name) {
    ++model_test::s.calls[std::string("find:") + name];
    if (model_test::s.omit_head && strcmp(name, "token_embd.weight") != 0) return nullptr;
    for (unsigned i = 0; i < 3; ++i) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return nullptr;
}

int qrt_gguf_read_f32(qrt_gguf * g, const char * name, std::vector<float> * out,
                      char * err, size_t cap) {
    const qrt_gguf_tensor * t = qrt_gguf_find(g, name);
    if (t == nullptr) {
        snprintf(err, cap, "%s is not in the tensor table", name);
        return -1;
    }
    if (t->type != 0) {
        snprintf(err, cap, "%s is not f32", name);
        return -1;
    }
    uint64_t count = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) count *= t->ne[i];
    out->assign((size_t) count, 0.0f);
    return qrt_gguf_read_tensor(g, t, out->data(), out->size() * sizeof(float), err, cap);
}

int qrt_gguf_read_tensor(qrt_gguf *, const qrt_gguf_tensor * t, void * dst, size_t bytes,
                         char * err, size_t cap) {
    if (model_test::hit("norm_read", err, cap) != 0) return -1;
    model_test::require(t->type == 0 && bytes == 128, "norm shape");
    float * p = static_cast<float *>(dst);
    for (unsigned i = 0; i < 32; ++i) p[i] = 1.0f + (float) i / 64.0f;
    return 0;
}

int qrt_gguf_read_range(qrt_gguf *, uint64_t offset, void * dst, size_t bytes,
                        char * err, size_t cap) {
    const bool head = offset >= 2048;
    if (model_test::hit(head ? "head_read" : "embed_read", err, cap) != 0) return -1;
    const uint64_t row = (offset - (head ? 2048u : 1024u)) / 34u;
    model_test::require(bytes == 34 && row < 3, "q8 row request");
    unsigned char * p = static_cast<unsigned char *>(dst);
    p[0] = 0;
    p[1] = 0x3c;
    for (unsigned i = 0; i < 32; ++i) {
        const int value = head ? (int) ((i * (row + 2) + 1) % 13) - 6 : (int) ((i + row * 3) % 9) - 4;
        p[i + 2] = (unsigned char) (int8_t) value;
    }
    return 0;
}

int qrt_layer_open(qrt_vk_dev *, const char *, const char *, uint32_t index, uint32_t tokens,
                   const char *, const qrt_layer_cfg * cfg, qrt_layer ** out,
                   char * err, size_t cap) {
    *out = nullptr;
    const int rc = model_test::hit("layer_open", err, cap);
    if (rc != 0 && !model_test::s.partial_create) return rc;
    qrt_layer * l = new qrt_layer;
    l->index = index;
    l->info.n_embd = 32;
    l->info.inter = 32;
    l->info.tokens = tokens;
    l->info.linear = index % 2 == 0 ? 1 : 0;
    l->input.resize((size_t) tokens * 32);
    l->output.values.resize(l->input.size());
    model_test::s.layer_configs.push_back(*cfg);
    model_test::s.layer_chunks.push_back(tokens);
    model_test::acquire("layer");
    *out = l;
    return rc;
}

void qrt_layer_close(qrt_vk_dev * dev, qrt_layer * l) {
    model_test::require(dev != nullptr && l != nullptr, "close layer with live device");
    model_test::release("layer");
    delete l;
}

const qrt_layer_info * qrt_layer_shape(const qrt_layer * l) { return &l->info; }
double qrt_layer_last_ms(const qrt_layer *) { return 0.0; }
VkBuffer qrt_layer_out(const qrt_layer * l) { return const_cast<VkBuffer_T *>(&l->output); }
VkBuffer qrt_layer_x(const qrt_layer * l) { return qrt_layer_out(l); }
VkBuffer qrt_layer_xnorm(const qrt_layer * l) { return qrt_layer_out(l); }
VkBuffer qrt_layer_attn_src(const qrt_layer * l) { return qrt_layer_out(l); }
VkBuffer qrt_layer_attn_out(const qrt_layer * l) { return qrt_layer_out(l); }
VkBuffer qrt_layer_post_norm(const qrt_layer * l) { return qrt_layer_out(l); }
uint64_t qrt_layer_uploaded_bytes(const qrt_layer *) { return 0; }
const qrt_ffn_bufs * qrt_layer_ffn_bufs(const qrt_layer *) { return nullptr; }

int qrt_layer_zero_state(qrt_layer * l, char * err, size_t cap) {
    if (model_test::hit("reset", err, cap) != 0) return -1;
    l->elapsed = 0;
    return 0;
}

int qrt_layer_set_input_rows(qrt_layer * l, const float * x, uint32_t rows,
                             char * err, size_t cap) {
    if (rows == 0 || rows > l->info.tokens) return -1;
    if (model_test::hit("set_input", err, cap) != 0) return -1;
    const size_t n = (size_t) rows * 32;
    std::copy(x, x + n, l->input.begin());
    if (!l->captured) {
        model_test::s.layer_inputs.push_back(l->input);
        l->captured = true;
    }
    return 0;
}

int qrt_layer_set_input(qrt_layer * l, const float * x, char * err, size_t cap) {
    return qrt_layer_set_input_rows(l, x, l->info.tokens, err, cap);
}

int qrt_vk_alloc(qrt_vk_dev *, uint64_t, VkBufferUsageFlags, VkBuffer * buf,
                 VkDeviceMemory * mem, char *, size_t) {
    *buf = VK_NULL_HANDLE;
    *mem = VK_NULL_HANDLE;
    return -1;
}

void qrt_vk_free(qrt_vk_dev *, VkBuffer, VkDeviceMemory) {}

int qrt_weights_open(const char *, qrt_weights ** out, char *, size_t) {
    *out = nullptr;
    return -1;
}

void qrt_weights_close(qrt_weights *) {}
uint32_t qrt_weights_experts(const qrt_weights *) { return 0; }
uint64_t qrt_weights_max_expert_bytes(const qrt_weights *) { return 0; }

int qrt_layer_run_chunk(qrt_layer * l, uint32_t pos, uint32_t tokens, char * err, size_t cap) {
    if (model_test::hit("run", err, cap) != 0) return -1;
    model_test::require(tokens <= l->info.tokens, "chunk length fits layer");
    model_test::s.positions.push_back(pos);
    model_test::s.lengths.push_back(tokens);
    for (uint32_t t = 0; t < tokens; ++t) {
        ++l->elapsed;
        for (unsigned i = 0; i < 32; ++i) {
            const size_t j = (size_t) t * 32 + i;
            l->output.values[j] = 2.0f * l->input[j] + (float) (l->index + 1 + l->elapsed * (i + 1));
        }
    }
    return 0;
}

int qrt_layer_ref_cpu(qrt_layer * l, const float * x, float * out, qrt_layer_ref_trace * tr,
                      int planes, char * err, size_t cap) {
    if (model_test::hit("reference", err, cap) != 0) return -1;
    model_test::s.ref_inputs.emplace_back(x, x + l->input.size());
    model_test::s.ref_planes.push_back(planes);
    for (uint32_t t = 0; t < l->info.tokens; ++t) {
        for (unsigned i = 0; i < 32; ++i) {
            const size_t j = (size_t) t * 32 + i;
            out[j] = 2.0f * x[j] + (float) (l->index + 1 + (t + 1) * (i + 1)) + model_test::s.ref_bias;
        }
    }
    float * traces[] = {tr->xnorm, tr->attn_src, tr->attn_out, tr->x_res, tr->post_norm};
    for (float * dst : traces) std::copy(out, out + l->input.size(), dst);
    return 0;
}

namespace model_test {

using entry = int (*)(const qrt_config *, const qrt_model_options *, const char *, const char *,
                     const char *, const uint32_t *, uint32_t, char *, size_t);

std::vector<uint32_t> tokens(uint32_t n) {
    std::vector<uint32_t> ids(n);
    for (uint32_t i = 0; i < n; ++i) ids[i] = (i * 7 + i / 3) % 3;
    return ids;
}

qrt_config config() {
    qrt_config cfg;
    qrt_config_defaults(&cfg);
    cfg.prefill_chunk = 64;
    cfg.kv_k = QRT_KV_F16;
    cfg.kv_v = QRT_KV_F16;
    cfg.kv_page_tokens = 96;
    cfg.vram_tier_mib = 1024;
    return cfg;
}

qrt_model_options options() {
    qrt_model_options opt;
    opt.n_layers = 2;
    opt.n_rows = 2;
    return opt;
}

int invoke(entry fn, const qrt_config & cfg, const qrt_model_options & opt,
           const std::vector<uint32_t> & ids, char * err, size_t cap) {
    return fn(&cfg, &opt, "model-test.gguf", "model-test-store", "model-test-spv",
              ids.data(), (uint32_t) ids.size(), err, cap);
}

void ids_case(const std::string & text, bool valid) {
    reset();
    s.files["model-test.ids"] = std::vector<unsigned char>(text.begin(), text.end());
    uint32_t sentinel = 7;
    uint32_t * ids = &sentinel;
    uint32_t count = 99;
    char err[256] = {};
    const int rc = qrt_model_read_ids("model-test.ids", &ids, &count, err, sizeof(err));
    require((rc == 0) == valid, "token parser verdict");
    if (valid) {
        require(count == 3 && ids[0] == 0 && ids[1] == 17 && ids[2] == INT32_MAX, "decimal token values");
        free(ids);
    } else {
        require(ids == nullptr && count == 0 && err[0] != '\0', "no partial tokens on error");
    }
    balanced();
}

void test_ids() {
    ids_case(" 0\t00017\r\n2147483647\v\f", true);
    ids_case("0 17 2147483647", true);
    const char * bad[] = {"", " \t\n", "0 17x", "0 1 garbage", "2147483648", "0 99999999999999999999999999999999",
                         "-1", "+1", "1.0", "0x10", "1e3", "0 1,2"};
    for (const char * text : bad) ids_case(text, false);
    ids_case(std::string("0 1\0 2", 7), false);
    ids_case(std::string("0 1\x1a 2", 7), false);
    for (unsigned fault = 0; fault < 3; ++fault) {
        reset();
        s.files["model-test.ids"] = {'0', ' ', '1', '7', ' ', '2'};
        s.read_error_at = fault == 0 ? 6 : 0;
        s.close_error = fault == 1;
        s.open_error = fault == 2;
        uint32_t * ids = nullptr;
        uint32_t count = 12;
        char err[256] = {};
        require(qrt_model_read_ids("model-test.ids", &ids, &count, err, sizeof(err)) != 0, "token IO failure");
        require(ids == nullptr && count == 0 && err[0] != '\0', "token IO failure is atomic");
        balanced();
    }
}

uint32_t word(const std::vector<unsigned char> & bytes, size_t offset) {
    require(offset + 4 <= bytes.size(), "dump word in bounds");
    return (uint32_t) bytes[offset] | ((uint32_t) bytes[offset + 1] << 8) |
           ((uint32_t) bytes[offset + 2] << 16) | ((uint32_t) bytes[offset + 3] << 24);
}

void test_dump() {
    const uint32_t ids[] = {9, 4, 7, 2};
    const float values[] = {1.25f, -2.0f, 77.0f, 3.5f, -4.25f, 88.0f};
    for (unsigned fault = 0; fault <= 7; ++fault) {
        reset();
        s.short_write_at = fault <= 5 ? fault : 0;
        s.close_error = fault == 6;
        s.open_error = fault == 7;
        char err[256] = {};
        const int rc = qrt_model_write_dump("model-test.dump", 2, 2, 1, ids, values, 3, err, sizeof(err));
        require((rc == 0) == (fault == 0), "dump write failures propagate");
        if (fault != 0) require(err[0] != '\0', "dump error has a message");
        if (fault >= 1 && fault <= 5) require(s.writes == fault, "stop immediately on short write");
        if (fault == 0) {
            const auto & bytes = s.files.at("model-test.dump");
            require(bytes.size() == 24 + 12 + 16, "QRTLOG1 byte count");
            require(memcmp(bytes.data(), "QRTLOG1\0", 8) == 0, "QRTLOG1 magic");
            require(word(bytes, 8) == 1 && word(bytes, 12) == 2 && word(bytes, 16) == 2 && word(bytes, 20) == 1,
                    "QRTLOG1 header");
            require(word(bytes, 24) == 4 && word(bytes, 28) == 7 && word(bytes, 32) == 2, "dump covered token range");
            require(word(bytes, 36) == 0x3fa00000 && word(bytes, 40) == 0xc0000000 &&
                    word(bytes, 44) == 0x40600000 && word(bytes, 48) == 0xc0880000, "dump floats and stride");
        }
        balanced();
    }
}

void test_failures() {
    const struct { const char * name; unsigned call; } faults[] = {
        {"dev_open", 1}, {"gguf_open", 1}, {"embed_read", 2}, {"norm_read", 1},
        {"staging_create", 1}, {"layer_open", 1}, {"layer_open", 2},
        {"reset", 1}, {"reset", 2}, {"set_input", 1}, {"set_input", 2},
        {"run", 1}, {"run", 2}, {"download", 1}, {"download", 2},
        {"download", 3}, {"reference", 1}, {"head_read", 2}
    };
    for (const auto & fault : faults) {
        for (bool partial : {false, true}) {
            reset();
            s.failure = fault.name;
            s.failure_call = fault.call;
            s.partial_create = partial;
            qrt_model_options opt = options();
            opt.ref_out_path = "model-test.ref";
            char err[256] = {};
            require(invoke(qrt_model_check, config(), opt, tokens(64), err, sizeof(err)) == -1, "injected failure returns error");
            require(strstr(err, fault.name) != nullptr, "injected error preserved");
            require(s.calls[fault.name] == fault.call, "failure is not swallowed or retried");
            require(s.file_opened == 0, "no dump after failed forward");
            balanced();
        }
    }
    reset();
    s.short_write_at = 4;
    qrt_model_options opt = options();
    opt.out_path = "model-test.dev";
    char err[256] = {};
    require(invoke(qrt_model_logits, config(), opt, tokens(64), err, sizeof(err)) == -1, "public dump failure");
    balanced();
}

void test_check_and_reseed() {
    for (int reseed = 0; reseed <= 1; ++reseed) {
        reset();
        qrt_model_options opt = options();
        opt.ref_out_path = "model-test.ref";
        opt.reseed = reseed;
        opt.device_planes = reseed;
        char err[256] = {};
        require(invoke(qrt_model_check, config(), opt, tokens(64), err, sizeof(err)) == 0, "ref-only check succeeds");
        require(s.calls["head_read"] == 3 && s.file_opened == 1, "ref-only still computes the two logits arms");
        require(s.ref_inputs == s.layer_inputs, "reference starts at pre-layer inputs");
        require(s.ref_planes == std::vector<int>({reseed, reseed}), "device_planes reaches reference");
        const auto & dump = s.files.at("model-test.ref");
        require(dump.size() == 24 + 12 + 24 && word(dump, 20) == 61, "public dump dimensions");
        balanced();
    }
    for (int reseed = 0; reseed <= 1; ++reseed) {
        reset();
        s.ref_bias = 0.25f;
        qrt_model_options opt = options();
        opt.ref_out_path = "model-test.ref";
        opt.reseed = reseed;
        char err[256] = {};
        require(invoke(qrt_model_check, config(), opt, tokens(64), err, sizeof(err)) >= 0, "biased reference completes");
        require(s.ref_inputs.size() == 2 && s.layer_inputs.size() == 2, "two reference layer inputs captured");
        require(s.ref_inputs[0] == s.layer_inputs[0], "reference first layer uses embedding");
        for (size_t i = 0; i < s.ref_inputs[1].size(); ++i) {
            const float expected = s.layer_inputs[1][i] + (reseed ? 0.0f : 0.25f);
            require(s.ref_inputs[1][i] == expected, "reseed isolates layers while independent arm carries itself");
        }
        balanced();
    }
}

void test_chunks_and_config() {
    reset();
    s.omit_head = true;
    qrt_config cfg = config();
    cfg.prefill_chunk = 64;
    cfg.kv_k = QRT_KV_Q8_0;
    cfg.kv_v = QRT_KV_Q4_0;
    char err[256] = {};
    require(invoke(qrt_model_chunk_check, cfg, options(), tokens(96), err, sizeof(err)) == 0,
            "carry matches single chunk and reset counterexample differs");
    require(s.calls["find:output.weight"] == 0 && s.calls["find:output_norm.weight"] == 0 &&
            s.calls["head_read"] == 0 && s.calls["norm_read"] == 0, "hidden-only never scans head");
    require(s.tiers == std::vector<uint32_t>({1024, 1024, 1024}), "tier reaches all devices");
    require(s.layer_chunks == std::vector<uint32_t>({96, 96, 64, 64, 64, 64}), "chunk reaches layer creation");
    require(s.positions == std::vector<uint32_t>({0, 0, 0, 64, 0, 64, 0, 64, 0, 64}), "chunk positions");
    require(s.lengths == std::vector<uint32_t>({96, 96, 64, 32, 64, 32, 64, 32, 64, 32}), "tail chunk length");
    for (const qrt_layer_cfg & c : s.layer_configs) {
        require(c.kv_k == cfg.kv_k && c.kv_v == cfg.kv_v && c.kv_page_tokens == 96 && c.n_ctx == 96,
                "KV precision, page size and context reach layers");
        require(c.ffn == 1 && c.dn_recur == 1 && c.single_submit == 1, "numeric switches retained");
    }
    balanced();
    reset();
    s.failure = "reset";
    s.failure_call = 6;
    require(invoke(qrt_model_chunk_check, cfg, options(), tokens(96), err, sizeof(err)) == -1,
            "no-carry reset failure propagates");
    require(strstr(err, "reset") != nullptr, "no-carry reset error preserved");
    balanced();
}

void test_validation() {
    for (unsigned c = 0; c < 18; ++c) {
        reset();
        qrt_config cfg = config();
        qrt_model_options opt = options();
        opt.ref_out_path = "model-test.ref";
        std::vector<uint32_t> ids = tokens(64);
        switch (c) {
            case 0: cfg.prefill_chunk = -1; break;
            case 1: cfg.kv_page_tokens = 33; break;
            case 2: cfg.vram_tier_mib = -1; break;
            case 3: cfg.prefill_chunk = 33; break;
            case 4: opt.n_rows = 0; break;
            case 5: opt.n_rows = UINT32_MAX; break;
            case 6: opt.ref_out_path = nullptr; break;
            case 7: cfg.kv_k = QRT_KV_Q8_0; break;
            case 8: cfg.kv_v = QRT_KV_Q4_0; break;
            case 9: cfg.prefill_chunk = 32; break;
            case 10: opt.out_path = "./model-test.ref"; break;
            case 11: opt.out_path = ""; break;
            case 12: opt.reseed = 2; break;
            case 13: opt.device_planes = -1; break;
            case 14: ids.pop_back(); break;
            case 15: opt.out_path = "model-test.gguf"; break;
            case 16: opt.out_path = "model-test-store.qwen36rt.idx"; break;
            case 17: opt.ref_out_path = "model-test-store.qwen36rt.bin"; break;
        }
        char err[256] = {};
        require(invoke(qrt_model_check, cfg, opt, ids, err, sizeof(err)) == -1, "invalid request rejected");
        require(s.acquired == 0 && s.file_opened == 0 && err[0] != '\0', "request validated before resources");
    }
    for (unsigned c = 0; c < 5; ++c) {
        reset();
        qrt_model_options opt = options();
        if (c == 0) opt.n_layers = 3;
        if (c == 1) opt.n_layers = UINT32_MAX;
        if (c == 2) s.hp.nextn = -1;
        if (c == 3) s.hp.nextn = 4;
        if (c == 4) s.hp.present &= ~QRT_HP_BLOCK_COUNT;
        char err[256] = {};
        require(invoke(qrt_model_logits, config(), opt, tokens(64), err, sizeof(err)) == -1, "main-layer bound rejected");
        require(s.calls["layer_open"] == 0 && s.calls["embed_read"] == 0, "MTP is not silently included");
        balanced();
    }
    reset();
    s.hp.present &= ~QRT_HP_NEXTN;
    qrt_model_options opt = options();
    opt.n_layers = 3;
    char err[256] = {};
    require(invoke(qrt_model_logits, config(), opt, tokens(64), err, sizeof(err)) == 0, "absent nextn means no MTP");
    balanced();
    reset();
    std::vector<uint32_t> ids = tokens(64);
    ids[7] = 3;
    require(invoke(qrt_model_logits, config(), options(), ids, err, sizeof(err)) == -1, "out-of-vocabulary token rejected");
    balanced();
    reset();
    require(qrt_model_logits(nullptr, &opt, "x", nullptr, "x", ids.data(), 64, err, sizeof(err)) == -1,
            "null config rejected");
    const qrt_config cfg = config();
    require(qrt_model_logits(&cfg, nullptr, "x", nullptr, "x", ids.data(), 64, err, sizeof(err)) == -1,
            "null options rejected");
    require(s.acquired == 0, "null request has no side effects");
    balanced();
}

} // namespace model_test

int main() {
    model_test::test_ids();
    model_test::test_dump();
    model_test::test_failures();
    model_test::test_check_and_reseed();
    model_test::test_chunks_and_config();
    for (uint32_t n : {65u, 66u, 67u, 95u}) {
        model_test::reset();
        char err[256] = {};
        model_test::require(model_test::invoke(qrt_model_chunk_check, model_test::config(),
                            model_test::options(), model_test::tokens(n), err, sizeof(err)) == 0,
                            "partial tail is not truncated or processed as padding");
        model_test::balanced();
    }
    model_test::test_validation();
    model_test::balanced();
    puts("model_test PASS");
    return 0;
}
