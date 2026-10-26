#pragma once

#include <stdint.h>
#include <vector>

#include "attn_deltanet.h"
#include "attn_full.h"
#include "ffn.h"
#include "kernels/gemm.h"
#include "kernels/norm.h"
#include "layer.h"
#include "vk_dev.h"

#ifdef QRT_HAVE_VULKAN

#include "vk_transfer.h"
#include "vk_pass.h"

// Module-private state shared by the production TU and the CPU reference/check TU.
// Nothing outside layer/ includes this header.

// One prepared q8_0 projection; raw keeps the canonical blocks for the CPU reference.
struct qrt_layer_proj {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint32_t n;
    std::vector<unsigned char> raw;
};

struct qrt_layer {
    qrt_vk_dev *    dev;
    qrt_layer_info  info;
    qrt_layer_cfg   cfg;
    uint32_t        layer;
    float           eps;
    float           l2_eps;
    qrt_gemm *      gemm;
    qrt_norm *      nrm;
    qrt_deltanet *  dn;
    qrt_dn_weights * dnw;
    qrt_full *      full;
    qrt_ffn *       ffn;
    qrt_vk_staging * stg;
    qrt_vk_cmd *    pass;
    qrt_dn_shapes   dsh;
    qrt_full_shapes fsh;

    uint32_t        n_proj;
    qrt_layer_proj  proj[4];
    VkBuffer        proj_out[4];

    VkBuffer        x, norm_out, xnorm, attn_src, attn_out, xpost, xout;
    VkDeviceMemory  x_mem, norm_mem, xnorm_mem, out_mem, post_mem, xout_mem;
    VkBuffer        qs, ds, p2_qs, p2_ds, norm_w, post_w;
    VkDeviceMemory  qs_mem, ds_mem, p2qs_mem, p2ds_mem, nw_mem, pw_mem;

    VkDescriptorSetLayout add_dsl;
    VkPipelineLayout add_layout;
    VkDescriptorPool add_pool;
    VkPipeline       add_pipe;
    VkCommandPool    add_cmd_pool;
    VkCommandBuffer  add_cmd;
    VkFence          add_fence;

    // host copies the check's reference needs
    std::vector<float> h_x, h_norm, h_post, h_qnw, h_knw;
    uint32_t           run_pos0;    // position of the chunk's first row, this run
    uint32_t           run_tokens;  // rows of the opened chunk that belong to the sequence
    uint32_t           dn_snap;     // 1 while the last pass kept a snapshot per row
    uint32_t           dn_carry_rows;  // rows whose last taps stage the next chunk's conv
    double             last_ms;
    double             pass1_ms;
    double             group_ms;
    double             pass2_ms;
    double             gpu_wait_ms;  // submit->fence wait of both passes: the device leg
    uint32_t           submits;
    uint32_t           dispatches;
};

#endif  // QRT_HAVE_VULKAN
