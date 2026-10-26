// M2-5 step 2: one routed expert from the repacked store, through the int8 gate/up leg,
// then a CPU dequant cross-check of the gate output (the down matrix is Q5_K in this
// store and still skipped). Each check command owns its device handle, so it lives in
// its own translation unit.

#include "qrt_checks.h"

#include "qrt_config.h"
#include "qrt_util.h"
#include "gguf_index.h"
#include "vk_dev.h"
#include "kernels/moe_expert.h"
#include "weights.h"

#include <stdio.h>
#include <string.h>

int qrt_check_moe(const qrt_config * cfg, const char * prefix, const char * spv_dir,
                         uint32_t layer, uint32_t expert, uint32_t rows) {
    char err[QRT_ERR_SIZE] = {0};
    qrt_vk_dev * dev = NULL;
    if (qrt_vk_open((uint32_t) cfg->vram_tier_mib, &dev, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        return 2;
    }
    printf("moe_check device_dot_product=%d\n", dev->has_dot_product);
    qrt_weights * w = NULL;
    if (qrt_weights_open(prefix, &w, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        qrt_vk_close(dev);
        return 2;
    }
    qrt_moe_cfg mcfg;
    memset(&mcfg, 0, sizeof(mcfg));
    mcfg.tokens = 64;        // 2048 tokens x top-8 over 256 experts
    mcfg.hidden = 2048;
    mcfg.ffn    = 512;
    qrt_moe * moe = NULL;
    int rc = 0;
    if (qrt_moe_open(dev, w, spv_dir, &mcfg, &moe, err, sizeof(err)) != 0) {
        fprintf(stderr, "qrt: %s\n", err);
        rc = 2;
    } else {
        static const char * type_name[] = { "q4k", "q5k", "q6k" };
        qrt_moe_check_result res;
        const double t0 = now_ms();
        if (qrt_moe_check(dev, moe, w, layer, expert, rows, &res, err, sizeof(err)) != 0) {
            fprintf(stderr, "qrt: %s\n", err);
            rc = 2;
        } else {
            const double dt = now_ms() - t0;
            const int pass = res.gate_rel_l1 <= 1e-4 && res.down_rel_l1 <= 1e-4 &&
                             res.chain_rel_l1 <= 1e-4 && res.batch_rel_l1 <= 1e-4;
            printf("moe_matrix_check layer=%u expert=%u rows=%u tokens=%u "
                   "gate=%s gate_rel_l1=%.3e down=%s down_rel_l1=%.3e "
                   "chain_rel_l1=%.3e batch_rel_l1=%.3e ms=%.1f verdict=%s\n",
                   layer, expert, rows, mcfg.tokens, type_name[res.gate_type],
                   res.gate_rel_l1, type_name[res.down_type], res.down_rel_l1,
                   res.chain_rel_l1, res.batch_rel_l1, dt, pass ? "PASS" : "FAIL");
            if (!pass) {
                rc = 1;
            }
        }
        qrt_moe_close(dev, moe);
    }
    qrt_weights_close(w);
    printf("moe_io_failures=%u alloc_mib=%llu\n", dev->io_failures,
           (unsigned long long) (dev->alloc_bytes >> 20));
    qrt_vk_close(dev);
    return rc;
}

