// qrt-proto.cpp - one "pass" of the qwen36 MoE runtime, without the model: reads the dense
// weights plus the k-token expert union from synthetic device buffers and times the pass.
// One command buffer is recorded per k at startup and submitted repeatedly, so the reported
// submit/sync cost is what a pre-recorded pass costs (design 4.6). Columns: see README.md.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static constexpr VkDeviceSize MIB = 1024ull * 1024ull;

// Pass geometry from design 2.1 / 6.1: 2.02 GiB dense per pass spread over 40 layers, 1.828 MiB
// per expert. Every k arm reads its union from the one buffer sized for the largest union.
static constexpr VkDeviceSize DENSE_BYTES  = 2068 * MIB;
static constexpr VkDeviceSize EXPERT_BYTES = 1916796;
static constexpr uint32_t     LAYERS       = 40;

struct Arm {
    uint32_t k;
    double   u; // unique experts per layer for k tokens (M1 trace)
};

static const Arm ARMS[] = {
    {1,  8.0},
    {2, 12.7},
    {4, 19.9},
    {5, 23.5},
    {8, 31.6},
};
static constexpr uint32_t N_ARMS = sizeof(ARMS) / sizeof(ARMS[0]);

struct PC {
    uint32_t base; // first vec4 of this layer's slice
    uint32_t n;    // vec4 count in the slice
};

static void die(const char * what, VkResult r) {
    std::printf("qrt_proto: %s failed, VkResult %d\n", what, (int) r);
    std::exit(2);
}

#define VK_TRY(x) do { const VkResult r_ = (x); if (r_ != VK_SUCCESS) { die(#x, r_); } } while (0)

static VkDeviceSize align16(VkDeviceSize v) {
    return (v + 15) & ~(VkDeviceSize) 15;
}

static VkDeviceSize align16_down(VkDeviceSize v) {
    return v & ~(VkDeviceSize) 15;
}

static VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
    return (v + a - 1) / a * a;
}

static double mib(VkDeviceSize b) {
    return (double) b / (double) MIB;
}

static double gb(VkDeviceSize b) {
    return (double) b / 1e9;
}

static std::vector<char> read_file(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::printf("qrt_proto: cannot open shader %s\n", path);
        std::exit(2);
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data((size_t) size);
    f.read(data.data(), size);
    return data;
}

using Clock = std::chrono::steady_clock;

static double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double median(const std::vector<double> & v) {
    std::vector<double> keep;
    for (double x : v) {
        if (x >= 0.0) {
            keep.push_back(x);
        }
    }
    if (keep.empty()) {
        return -1.0;
    }
    std::sort(keep.begin(), keep.end());
    return keep[keep.size() / 2];
}

struct Buf {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
};

struct Ctx {
    VkInstance                       inst = VK_NULL_HANDLE;
    VkPhysicalDevice                 phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties       props{};
    VkPhysicalDeviceMemoryProperties memprops{};
    VkDevice                         dev = VK_NULL_HANDLE;
    uint32_t                         fam = 0;
    uint32_t                         ts_bits = 0;
    VkQueue                          q = VK_NULL_HANDLE;
    VkCommandPool                    pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout            dsl = VK_NULL_HANDLE;
    VkPipelineLayout                 pl = VK_NULL_HANDLE;
    VkPipeline                       pipe = VK_NULL_HANDLE;
    VkDescriptorPool                 dpool = VK_NULL_HANDLE;
    VkSemaphore                      sem = VK_NULL_HANDLE;
    uint64_t                         seq = 0;
    VkQueryPool                      qpool = VK_NULL_HANDLE;
};

static Buf make_buf(Ctx & c, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_TRY(vkCreateBuffer(c.dev, &bci, nullptr, &b.buf));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(c.dev, b.buf, &req);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < c.memprops.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = c.memprops.memoryTypes[i].propertyFlags;
        if ((req.memoryTypeBits & (1u << i)) && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) {
        std::printf("qrt_proto: no device-local memory type for a %.1f MiB buffer\n", mib(size));
        std::exit(2);
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = type;
    const VkResult r = vkAllocateMemory(c.dev, &mai, nullptr, &b.mem);
    if (r != VK_SUCCESS) {
        std::printf("qrt_proto: cannot allocate %.1f MiB, VkResult %d (close other GPU users)\n",
                mib(size), (int) r);
        std::exit(2);
    }
    VK_TRY(vkBindBufferMemory(c.dev, b.buf, b.mem, 0));
    return b;
}

static VkPipeline make_pipeline(Ctx & c, const char * spv_path) {
    const std::vector<char> code = read_file(spv_path);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode    = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_TRY(vkCreateShaderModule(c.dev, &smci, nullptr, &mod));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = c.pl;
    VkPipeline p = VK_NULL_HANDLE;
    VK_TRY(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
    vkDestroyShaderModule(c.dev, mod, nullptr);
    return p;
}

static void init_ctx(Ctx & c, const char * spv_path) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "qrt_proto";
    app.apiVersion       = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_TRY(vkCreateInstance(&ici, nullptr, &c.inst));

    uint32_t n = 0;
    VK_TRY(vkEnumeratePhysicalDevices(c.inst, &n, nullptr));
    if (n == 0) {
        std::printf("qrt_proto: no Vulkan device\n");
        std::exit(2);
    }
    std::vector<VkPhysicalDevice> devs(n);
    VK_TRY(vkEnumeratePhysicalDevices(c.inst, &n, devs.data()));
    int pick = -1;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        std::printf("# gpu[%u] %s type=%d\n", i, p.deviceName, (int) p.deviceType);
        if (pick < 0 && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            pick = (int) i;
        }
    }
    if (pick < 0) {
        pick = 0;
    }
    c.phys = devs[(size_t) pick];
    vkGetPhysicalDeviceProperties(c.phys, &c.props);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.memprops);

    uint32_t nf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nf, nullptr);
    std::vector<VkQueueFamilyProperties> fams(nf);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nf, fams.data());
    int compute_fam = -1;
    for (uint32_t i = 0; i < nf; ++i) {
        const bool has_c = (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool has_g = (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        // mirror ggml-vulkan: prefer a compute family that is not the graphics family
        if (has_c && !has_g) {
            compute_fam = (int) i;
            break;
        }
        if (compute_fam < 0 && has_c) {
            compute_fam = (int) i;
        }
    }
    if (compute_fam < 0) {
        std::printf("qrt_proto: no compute queue family\n");
        std::exit(2);
    }
    c.fam     = (uint32_t) compute_fam;
    c.ts_bits = fams[(size_t) compute_fam].timestampValidBits;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(c.phys, &f2);
    if (!f12.timelineSemaphore) {
        std::printf("qrt_proto: timeline semaphores are required (design 4.6)\n");
        std::exit(2);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.fam;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enable12.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext                = &enable12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    VK_TRY(vkCreateDevice(c.phys, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.fam, 0, &c.q);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = c.fam;
    VK_TRY(vkCreateCommandPool(c.dev, &cpci, nullptr, &c.pool));

    VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue  = 0;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &stci;
    VK_TRY(vkCreateSemaphore(c.dev, &sci, nullptr, &c.sem));

    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding         = 0;
    b[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[0].descriptorCount = 1;
    b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1] = b[0];
    b[1].binding = 1;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2;
    dslci.pBindings    = b;
    VK_TRY(vkCreateDescriptorSetLayout(c.dev, &dslci, nullptr, &c.dsl));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(PC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &c.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VK_TRY(vkCreatePipelineLayout(c.dev, &plci, nullptr, &c.pl));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 4;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    VK_TRY(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &c.dpool));

    c.pipe = make_pipeline(c, spv_path);

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;
    VK_TRY(vkCreateQueryPool(c.dev, &qpci, nullptr, &c.qpool));
}

static VkCommandBuffer alloc_cb(Ctx & c) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = c.pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VK_TRY(vkAllocateCommandBuffers(c.dev, &cai, &cb));
    return cb;
}

// The dense and expert slices share one allocation: two multi-GiB blocks can fail to fit on an
// 8 GiB card even when their sum does.
static VkDescriptorSet make_set(Ctx & c, VkBuffer src, VkDeviceSize src_off, VkDeviceSize src_size,
        VkBuffer dst) {
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = c.dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &c.dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_TRY(vkAllocateDescriptorSets(c.dev, &ai, &set));
    VkDescriptorBufferInfo bi[2] = {{src, src_off, src_size}, {dst, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet          = set;
        w[i].dstBinding      = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(c.dev, 2, w, 0, nullptr);
    return set;
}

// Each layer is one dense dispatch plus one expert-union dispatch; layers are separated by a
// barrier because the real pass reads the previous layer's output.
static void record_pass(Ctx & c, VkCommandBuffer cb, VkDescriptorSet set_dense,
        VkDescriptorSet set_expert, uint32_t dense_vec4, uint32_t expert_vec4, uint32_t groups,
        bool timestamps) {
    if (timestamps) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c.qpool, 0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipe);
    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
        PC pc;
        pc.base = layer * dense_vec4;
        pc.n    = dense_vec4;
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &set_dense, 0, nullptr);
        vkCmdPushConstants(cb, c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatch(cb, groups, 1, 1);

        pc.base = layer * expert_vec4;
        pc.n    = expert_vec4;
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &set_expert, 0, nullptr);
        vkCmdPushConstants(cb, c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatch(cb, groups, 1, 1);

        VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar, 0, nullptr, 0, nullptr);
    }
    if (timestamps) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c.qpool, 1);
    }
}

static uint64_t submit_pass(Ctx & c, VkCommandBuffer cb) {
    const uint64_t value = ++c.seq;
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues    = &value;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext                = &ts;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &c.sem;
    VK_TRY(vkQueueSubmit(c.q, 1, &si, VK_NULL_HANDLE));
    return value;
}

static void wait_pass(Ctx & c, uint64_t value) {
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores    = &c.sem;
    wi.pValues        = &value;
    VK_TRY(vkWaitSemaphores(c.dev, &wi, UINT64_MAX));
}

int main(int argc, char ** argv) {
    const uint32_t reps   = argc > 1 ? (uint32_t) std::atoi(argv[1]) : 20;
    const uint32_t warmup = argc > 2 ? (uint32_t) std::atoi(argv[2]) : 3;
    const uint32_t groups = argc > 3 ? (uint32_t) std::atoi(argv[3]) : 1024;
    const char *   spv    = argc > 4 ? argv[4] : "qrt-read.spv";
    if (reps == 0 || groups == 0) {
        std::printf("usage: qrt_proto [reps] [warmup] [groups] [shader.spv]\n");
        return 1;
    }

    Ctx c;
    init_ctx(c, spv);
    if (groups > c.props.limits.maxComputeWorkGroupCount[0]) {
        std::printf("qrt_proto: groups=%u exceeds maxComputeWorkGroupCount %u\n",
                groups, c.props.limits.maxComputeWorkGroupCount[0]);
        return 2;
    }

    const VkDeviceSize dense_layer = align16_down(DENSE_BYTES / LAYERS);
    VkDeviceSize expert_layer[N_ARMS];
    VkDeviceSize expert_max = 0;
    for (uint32_t i = 0; i < N_ARMS; ++i) {
        expert_layer[i] = align16((VkDeviceSize) (ARMS[i].u * (double) EXPERT_BYTES + 0.5));
        expert_max      = std::max(expert_max, expert_layer[i]);
    }

    const VkDeviceSize dense_bytes  = dense_layer * LAYERS;
    const VkDeviceSize expert_bytes = expert_max * LAYERS;
    const VkDeviceSize expert_off   = align_up(dense_bytes, 256);
    const Buf pool = make_buf(c, expert_off + expert_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const Buf out  = make_buf(c, 4096 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    const VkDescriptorSet set_dense  = make_set(c, pool.buf, 0, dense_bytes, out.buf);
    const VkDescriptorSet set_expert = make_set(c, pool.buf, expert_off, expert_bytes, out.buf);

    VkCommandBuffer cb[N_ARMS];
    for (uint32_t i = 0; i < N_ARMS; ++i) {
        cb[i] = alloc_cb(c);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_TRY(vkBeginCommandBuffer(cb[i], &bi));
        record_pass(c, cb[i], set_dense, set_expert, (uint32_t) (dense_layer / 16),
                (uint32_t) (expert_layer[i] / 16), groups, c.ts_bits != 0);
        VK_TRY(vkEndCommandBuffer(cb[i]));
    }

    const VkDeviceSize compute_bytes = pool.size + out.size;
    std::printf("# qrt_proto device=%s groups=%u reps=%u warmup=%u timestamps=%s\n",
            c.props.deviceName, groups, reps, warmup, c.ts_bits != 0 ? "on" : "off");
    std::printf("# dense %.1f MiB (%u layers x %.1f MiB), expert %.0f B, buffers %.1f MiB\n",
            mib(dense_bytes), LAYERS, mib(dense_layer), (double) EXPERT_BYTES, mib(compute_bytes));
    std::printf("k,u_experts,expert_mib,read_mib,compute_mib,reps,submit_ms,sync_ms,pass_ms,gpu_ms,gbs\n");

    for (uint32_t a = 0; a < N_ARMS; ++a) {
        for (uint32_t w = 0; w < warmup; ++w) {
            const uint64_t v = submit_pass(c, cb[a]);
            wait_pass(c, v);
        }

        std::vector<double> sub(reps), syn(reps), tot(reps), gpu(reps, -1.0);
        for (uint32_t r = 0; r < reps; ++r) {
            if (c.ts_bits != 0) {
                vkResetQueryPool(c.dev, c.qpool, 0, 2);
            }
            const Clock::time_point t0 = Clock::now();
            const uint64_t v = submit_pass(c, cb[a]);
            const Clock::time_point t1 = Clock::now();
            wait_pass(c, v);
            const Clock::time_point t2 = Clock::now();

            sub[r] = ms_between(t0, t1);
            syn[r] = ms_between(t1, t2);
            tot[r] = ms_between(t0, t2);
            if (c.ts_bits != 0) {
                uint64_t ts[2] = {0, 0};
                const VkResult qr = vkGetQueryPoolResults(c.dev, c.qpool, 0, 2, sizeof(ts), ts,
                        sizeof(ts[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                if (qr == VK_SUCCESS && ts[1] >= ts[0]) {
                    gpu[r] = (double) (ts[1] - ts[0]) * (double) c.props.limits.timestampPeriod / 1e6;
                }
            }
        }

        const VkDeviceSize read_bytes = (dense_layer + expert_layer[a]) * LAYERS;
        const double       pass_ms   = median(tot);
        const double       gbs       = pass_ms > 0.0 ? gb(read_bytes) / (pass_ms / 1000.0) : 0.0;
        std::printf("%u,%.1f,%.1f,%.1f,%.1f,%u,%.3f,%.3f,%.3f,%.3f,%.1f\n",
                ARMS[a].k, ARMS[a].u, mib(expert_layer[a] * LAYERS), mib(read_bytes),
                mib(compute_bytes), reps, median(sub), median(syn), pass_ms, median(gpu), gbs);
        std::fflush(stdout);
    }

    vkDeviceWaitIdle(c.dev);
    return 0;
}
