// qrt - the qwen36 MoE runtime. M2-0 scaffold: the design section 7 switches, the
// telemetry sink and a selftest. No model, no Vulkan device, no engine yet.

#include "qrt_config.h"
#include "qrt_telemetry.h"
#include "gguf_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

static const char QRT_VERSION[] = "0.1-m2-0";

static int g_failures = 0;

static void expect(int ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++g_failures;
    }
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value != NULL ? value : "");
#else
    if (value != NULL) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// qrt_config_parse wants char **; every argument it reads is a literal or argv.
static char * lit(const char * s) {
    return const_cast<char *>(s);
}

static int parse_argv(qrt_config * cfg, int n, const char * const * args, char * err, size_t err_size) {
    char * av[24];
    if (n > 22) {
        return -1;
    }
    av[0] = lit("qrt");
    for (int i = 0; i < n; ++i) {
        av[i + 1] = lit(args[i]);
    }
    av[n + 1] = NULL;
    return qrt_config_parse(cfg, n + 1, av, err, err_size);
}

static int count_lines(const char * path, char * first, size_t first_size) {
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    int lines = 0;
    char buf[512];
    while (fgets(buf, sizeof(buf), f) != NULL) {
        if (lines == 0 && first != NULL && first_size > 0) {
            snprintf(first, first_size, "%s", buf);
        }
        ++lines;
    }
    fclose(f);
    return lines;
}

static int qrt_selftest(void) {
    char err[QRT_ERR_SIZE];

    // The selftest must not depend on the caller's environment.
    for (int i = 0; i < qrt_config_switch_count(); ++i) {
        set_env(qrt_config_env_name(i), NULL);
    }

    qrt_config d;
    qrt_config_defaults(&d);
    expect(d.spec_k == 4 && d.draft_vocab == 16384 && d.expert_cache_mib == 4198 &&
           d.miss_split == QRT_MISS_SPLIT_AUTO && d.cache_policy == QRT_CACHE_SLRU &&
           d.prefill_chunk == 2048 && d.kv_precision == QRT_KV_Q4_0 &&
           d.kv_page_tokens == 16384 && d.telemetry == 0,
           "defaults match design section 7");

    {
        const char * all[] = {
            "--spec-k", "5", "--draft-vocab", "32768", "--expert-cache-mib", "3800",
            "--miss-split", "60", "--cache-policy", "lru", "--prefill-chunk", "1024",
            "--kv-precision", "f16", "--kv-page-tokens", "8192", "--telemetry", "1",
            "--telemetry-path", "bench/raw/x.csv",
        };
        qrt_config c;
        qrt_config_defaults(&c);
        const int ok = parse_argv(&c, 20, all, err, sizeof(err)) == 0;
        expect(ok && c.spec_k == 5 && c.draft_vocab == 32768 && c.expert_cache_mib == 3800 &&
               c.miss_split == 60 && c.cache_policy == QRT_CACHE_LRU && c.prefill_chunk == 1024 &&
               c.kv_precision == QRT_KV_F16 && c.kv_page_tokens == 8192 && c.telemetry == 1 &&
               strcmp(c.telemetry_path, "bench/raw/x.csv") == 0,
               "all ten switches parse from the CLI");
    }

    {
        const char * eq[] = { "--spec-k=2", "--miss-split=auto" };
        qrt_config c;
        qrt_config_defaults(&c);
        expect(parse_argv(&c, 2, eq, err, sizeof(err)) == 0 && c.spec_k == 2 &&
               c.miss_split == QRT_MISS_SPLIT_AUTO,
               "flag=value form and the auto miss split");
    }

    {
        set_env("QRT_SPEC_K", "2");
        set_env("QRT_DRAFT_VOCAB", "0");
        qrt_config c;
        qrt_config_defaults(&c);
        const int ok = parse_argv(&c, 0, NULL, err, sizeof(err)) == 0;
        expect(ok && c.spec_k == 2 && c.draft_vocab == 0, "environment fills in when the CLI is silent");

        const char * over[] = { "--spec-k", "8" };
        expect(parse_argv(&c, 2, over, err, sizeof(err)) == 0 && c.spec_k == 8,
               "the CLI overrides the environment");
        set_env("QRT_SPEC_K", NULL);
        set_env("QRT_DRAFT_VOCAB", NULL);
    }

    {
        const char * bad[][2] = {
            { "--spec-k", "3" },        { "--draft-vocab", "12" },
            { "--expert-cache-mib", "9000" }, { "--miss-split", "101" },
            { "--cache-policy", "foo" }, { "--prefill-chunk", "0" },
            { "--kv-precision", "q8_0" }, { "--kv-page-tokens", "1000" },
            { "--telemetry", "2" },     { "--nope", "1" },
        };
        const int n_bad = (int) (sizeof(bad) / sizeof(bad[0]));
        int rejected = 0;
        for (int i = 0; i < n_bad; ++i) {
            qrt_config c;
            qrt_config_defaults(&c);
            char e[QRT_ERR_SIZE] = {0};
            const int ok = parse_argv(&c, 2, bad[i], e, sizeof(e)) != 0 && e[0] != '\0' && c.spec_k == 4;
            if (ok) {
                ++rejected;
            } else {
                printf("  not rejected: %s %s\n", bad[i][0], bad[i][1]);
            }
        }
        expect(rejected == n_bad, "every bad value is rejected with a message and leaves the config usable");
    }

    {
        const char * missing[] = { "--spec-k" };
        qrt_config c;
        qrt_config_defaults(&c);
        expect(parse_argv(&c, 1, missing, err, sizeof(err)) != 0 && c.spec_k == 4,
               "a flag without a value is a recoverable error");
    }

    {
        qrt_config c;
        qrt_config_defaults(&c);
        c.spec_k = 3;
        expect(qrt_config_validate(&c, err, sizeof(err)) != 0, "validate rejects a programmatic spec_k");
    }

    {
        const char * low[] = { "--prefill-chunk", "512" };
        qrt_config c;
        qrt_config_defaults(&c);
        const int ok = parse_argv(&c, 2, low, err, sizeof(err)) == 0 &&
                       qrt_config_validate(&c, err, sizeof(err)) == 0;
        FILE * sink = tmpfile();
        const int notes = sink != NULL ? qrt_config_notes(&c, sink) : 0;
        if (sink != NULL) {
            fclose(sink);
        }
        expect(ok && notes >= 1, "a legal but costly value is a note, not an error");
    }

    {
        const char * path = "qrt-selftest-telemetry.csv";
        remove(path);
        char e[QRT_ERR_SIZE] = {0};
        qrt_telemetry * t = qrt_telemetry_open(path, e, sizeof(e));
        expect(t != NULL, "the telemetry sink opens");

        qrt_pass_stats s;
        memset(&s, 0, sizeof(s));
        s.k = 4;
        s.pos = 100;
        s.accepted = 3;
        s.hit_experts = 30;
        s.miss_experts = 10;
        s.miss_cpu = 4;
        s.miss_h2d = 6;
        s.h2d_bytes = 6ull * 1916796ull;
        s.draft_ms = 0.6;
        s.gpu_ms = 18.2;
        s.cpu_leg_ms = 0.4;
        s.pass_ms = 19.1;
        qrt_telemetry_pass(t, 7, &s);
        qrt_telemetry_pass(t, 8, &s);
        expect(t != NULL && t->rows == 2 && qrt_telemetry_failed(t) == 0,
               "two pass rows are written and flushed");
        qrt_telemetry_close(t);

        char first[256] = {0};
        const int lines = count_lines(path, first, sizeof(first));
        expect(lines == 3 && strncmp(first, "pass,pos,k,accepted", 19) == 0,
               "the CSV holds the header and one row per pass");
        remove(path);

        qrt_telemetry_pass(NULL, 0, &s);
        qrt_telemetry_close(NULL);
        expect(1, "a NULL sink is a no-op");

        char e2[QRT_ERR_SIZE] = {0};
        expect(qrt_telemetry_open("", e2, sizeof(e2)) == NULL && e2[0] != '\0',
               "an unusable sink path is a recoverable error");
    }

    printf("\nselftest: %d failed\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    int print_config = 0;
    const char * gguf_path = NULL;
    const char * require_tensor = NULL;

    std::vector<char *> args;
    args.push_back(argc > 0 ? argv[0] : lit("qrt"));
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            qrt_config_print_switches(stdout);
            return 0;
        }
        if (strcmp(argv[i], "--selftest") == 0) {
            return qrt_selftest();
        }
        if (strcmp(argv[i], "--print-config") == 0) {
            print_config = 1;
            continue;
        }
        const int is_index = strcmp(argv[i], "--gguf-index") == 0;
        const int is_require = strcmp(argv[i], "--require-tensor") == 0;
        if (is_index || is_require) {
            if (i + 1 >= argc) {
                fprintf(stderr, "qrt: %s needs a value\n", argv[i]);
                return 2;
            }
            if (is_index) {
                gguf_path = argv[++i];
            } else {
                require_tensor = argv[++i];
            }
            continue;
        }
        args.push_back(argv[i]);
    }
    args.push_back(NULL);

    qrt_config cfg;
    qrt_config_defaults(&cfg);
    char err[QRT_ERR_SIZE] = {0};
    if (qrt_config_parse(&cfg, (int) args.size() - 1, args.data(), err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        fprintf(stderr, "qrt: run 'qrt --help' for the switch table; nothing was allocated or opened\n");
        return 2;
    }
    if (qrt_config_validate(&cfg, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    qrt_config_notes(&cfg, stderr);

    if (print_config) {
        qrt_config_print_effective(&cfg, stdout);
        return 0;
    }

    if (gguf_path != NULL) {
        qrt_gguf * g = NULL;
        if (qrt_gguf_open(gguf_path, &g, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        if (require_tensor != NULL) {
            const qrt_gguf_tensor * t = NULL;
            if (qrt_gguf_require(g, require_tensor, &t, err, sizeof(err)) != 0) {
                fprintf(stderr, "qrt: %s\n", err);
                qrt_gguf_close(g);
                return 2;
            }
            printf("tensor=%s offset=%llu nbytes=%llu type=%s\n", t->name,
                   (unsigned long long) t->offset, (unsigned long long) t->nbytes,
                   qrt_gguf_type_name(t->type));
        }
        qrt_gguf_print_report(g, stdout);
        qrt_gguf_close(g);
        return 0;
    }
    if (require_tensor != NULL) {
        fprintf(stderr, "qrt: --require-tensor needs --gguf-index <path>\n");
        return 2;
    }

    if (cfg.telemetry != 0) {
        qrt_telemetry * t = qrt_telemetry_open(cfg.telemetry_path, err, sizeof(err));
        if (t == NULL) {
            fprintf(stderr, "qrt: %s\n", err);
            return 2;
        }
        qrt_telemetry_close(t);
        printf("telemetry=%s (header written)\n", cfg.telemetry_path);
    }

    qrt_config_print_effective(&cfg, stdout);
    printf("qrt %s: switches parsed, no engine yet (M2-1 gguf_index is next)\n", QRT_VERSION);
    return 0;
}
