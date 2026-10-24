// q4k-gemm.cpp - R1c probe: effective Q4_K (weights) x Q8_1 (activations) int8 dot
// throughput of one prefill-shaped expert gate matrix, N rows x K, over M tokens.
// Weights and activations are generated on the host, uploaded once, then `reps` dispatches
// are timed behind one timeline semaphore. Columns, tolerances and limits: README.md.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "q4k-tiles.h"

// Byte-identical mirrors of the ggml block layouts, so the CPU reference reads exactly the
// bytes the shader reads: block_q4_K in ggml/src/ggml-common.h:337 (K_SCALE_SIZE = 12) and
// block_q8_1 in ggml/src/ggml-common.h:268.
struct q4k_block {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(q4k_block) == 144, "block_q4_K is 144 bytes");

struct q8_1_block {
    uint16_t d;
    uint16_t s; // d * sum(qs)
    int8_t   qs[32];
};
static_assert(sizeof(q8_1_block) == 36, "block_q8_1 is 36 bytes");

static constexpr uint32_t QK_K       = Q4K_QK;
static constexpr uint32_t QK8_1      = 32;
static constexpr uint32_t BLOCKS_PER_SB = QK_K / QK8_1;

// Per-element tolerance on the L1-normalized error (|err| / sum|terms|). The fp16 `s` field
// of block_q8_1 leaves ~1e-5 at these shapes; an index mapping bug is O(1).
static constexpr double TOL_REL = 1e-4;

static void die_msg(const char * msg) {
    std::printf("q4k_gemm: %s\n", msg);
    std::exit(2);
}

static void die(const char * what, VkResult r) {
    std::printf("q4k_gemm: %s failed, VkResult %d\n", what, (int) r);
    std::exit(2);
}

#define VK_TRY(x) do { const VkResult r_ = (x); if (r_ != VK_SUCCESS) { die(#x, r_); } } while (0)

// ---------------------------------------------------------------- fp16

// Round-to-nearest-even fp32 -> binary16. Values here are generated in a normal range, the
// subnormal and overflow branches only keep the generator honest.
static uint16_t f32_to_f16(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t mag  = x & 0x7FFFFFFFu;
    if (mag >= 0x7F800000u) {
        return (uint16_t) (sign | (mag > 0x7F800000u ? 0x7E00u : 0x7C00u));
    }
    if (mag >= 0x47800000u) {
        return (uint16_t) (sign | 0x7C00u);
    }
    if (mag < 0x38800000u) {
        if (mag < 0x33000000u) {
            return (uint16_t) sign;
        }
        const uint32_t exp   = mag >> 23;
        const uint32_t mant  = (mag & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126u - exp;
        uint32_t q = mant >> shift;
        const uint32_t rem  = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (q & 1u) != 0u)) {
            q++;
        }
        return (uint16_t) (sign | q);
    }
    uint32_t mant = mag & 0x7FFFFFu;
    uint32_t exp  = ((mag >> 23) - 127u) + 15u;
    const uint32_t rem = mant & 0x1FFFu;
    mant >>= 13;
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u) != 0u)) {
        mant++;
        if (mant == 0x400u) {
            mant = 0;
            exp++;
            if (exp >= 31u) {
                return (uint16_t) (sign | 0x7C00u);
            }
        }
    }
    return (uint16_t) (sign | (exp << 10) | mant);
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else { // subnormal: normalize
            uint32_t e = 127u - 15u + 1u;
            uint32_t m = mant;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                e--;
            }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// ---------------------------------------------------------------- deterministic generator

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1ull) {}
    uint32_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return (uint32_t) ((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * (float) (next() >> 8) / 16777216.0f;
    }
};

// Valid but random Q4_K: any 6-bit pair is decodable, d/dmin stay finite and positive.
static void gen_weights(std::vector<q4k_block> & blocks, uint32_t n, uint32_t nk, Rng & rng) {
    blocks.assign((size_t) n * nk, q4k_block{});
    for (size_t i = 0; i < blocks.size(); ++i) {
        q4k_block & b = blocks[i];
        b.d    = f32_to_f16(rng.uniform(1.0e-3f, 8.0e-3f));
        b.dmin = f32_to_f16(rng.uniform(1.0e-3f, 1.6e-2f));
        for (uint32_t j = 0; j < 12; ++j) {
            b.scales[j] = (uint8_t) (rng.next() & 0xFFu);
        }
        for (uint32_t j = 0; j < 128; ++j) {
            b.qs[j] = (uint8_t) (rng.next() & 0xFFu);
        }
    }
}

// Valid Q8_1: int8 quants, d = delta, s = d * sum(qs) as ggml stores it.
static void gen_acts(std::vector<q8_1_block> & blocks, uint32_t m, uint32_t nk32, Rng & rng) {
    blocks.assign((size_t) m * nk32, q8_1_block{});
    for (size_t i = 0; i < blocks.size(); ++i) {
        q8_1_block & b = blocks[i];
        int32_t sum = 0;
        for (uint32_t j = 0; j < 32; ++j) {
            const int32_t q = (int32_t) (rng.next() % 255u) - 127;
            b.qs[j] = (int8_t) q;
            sum += q;
        }
        const float d = rng.uniform(5.0e-3f, 2.0e-2f);
        b.d = f32_to_f16(d);
        b.s = f32_to_f16(f16_to_f32(b.d) * (float) sum);
    }
}

// ---------------------------------------------------------------- vulkan plumbing

struct Buf {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
};

struct Pc {
    uint32_t n;
    uint32_t k;
    uint32_t m;
};

struct Ctx {
    VkInstance                       inst = VK_NULL_HANDLE;
    VkPhysicalDevice                 phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties       props{};
    VkPhysicalDeviceMemoryProperties mem_props{};
    VkDevice                         dev = VK_NULL_HANDLE;
    uint32_t                         fam = 0;
    VkQueue                          queue = VK_NULL_HANDLE;
    VkCommandPool                    pool = VK_NULL_HANDLE;
    VkSemaphore                      timeline = VK_NULL_HANDLE;
    uint64_t                         seq = 0;
    bool                             dot_enabled = false;
    bool                             dot_feature = false;
    bool                             dot_packed_signed_fast = false;
    bool                             dot_via_core13 = false;
    VkDescriptorSetLayout            dsl = VK_NULL_HANDLE;
    VkPipelineLayout                 pl = VK_NULL_HANDLE;
    VkPipeline                       pipe = VK_NULL_HANDLE;
    VkDescriptorPool                 dpool = VK_NULL_HANDLE;
};

static bool has_device_ext(VkPhysicalDevice phys, const char * name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data());
    for (uint32_t i = 0; i < n; ++i) {
        if (std::strcmp(exts[i].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

static std::string version_str(uint32_t v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%u.%u.%u", VK_API_VERSION_MAJOR(v),
            VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
    return std::string(buf);
}

static uint32_t instance_api_version() {
    uint32_t have = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion fn =
            (PFN_vkEnumerateInstanceVersion) vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
    if (fn != nullptr && fn(&have) != VK_SUCCESS) {
        have = VK_API_VERSION_1_0;
    }
    if (have > VK_API_VERSION_1_3) {
        have = VK_API_VERSION_1_3;
    }
    return have;
}

static int pick_device(const std::vector<VkPhysicalDevice> & devs, int want) {
    if (want >= 0) {
        if ((size_t) want >= devs.size()) {
            std::printf("q4k_gemm: --dev %d out of range, %zu device(s)\n", want, devs.size());
            std::exit(1);
        }
        return want;
    }
    for (size_t i = 0; i < devs.size(); ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            return (int) i;
        }
    }
    return 0;
}

static void init_ctx(Ctx & c, int dev_index, bool force_fallback, bool list_only) {
    const uint32_t api = instance_api_version();
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "q4k_gemm";
    app.apiVersion       = api;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_TRY(vkCreateInstance(&ici, nullptr, &c.inst));

    uint32_t n = 0;
    VK_TRY(vkEnumeratePhysicalDevices(c.inst, &n, nullptr));
    if (n == 0) {
        die_msg("no Vulkan device");
    }
    std::vector<VkPhysicalDevice> devs(n);
    VK_TRY(vkEnumeratePhysicalDevices(c.inst, &n, devs.data()));
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        std::printf("# gpu[%u] %s type=%d api=%s\n", i, p.deviceName, (int) p.deviceType,
                version_str(p.apiVersion).c_str());
    }
    const int pick = pick_device(devs, dev_index);
    c.phys = devs[(size_t) pick];
    vkGetPhysicalDeviceProperties(c.phys, &c.props);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mem_props);

    if (list_only) {
        std::printf("# picked gpu[%d] by default (--dev %d to override)\n", pick, pick);
        std::exit(0);
    }

    if (c.props.apiVersion < VK_API_VERSION_1_2) {
        std::printf("q4k_gemm: device api %s is below the 1.2 floor (timeline semaphores)\n",
                version_str(c.props.apiVersion).c_str());
        std::exit(3);
    }

    // Integer dot: core in 1.3, else the KHR extension; query the same struct either way.
    const bool core13  = c.props.apiVersion >= VK_API_VERSION_1_3;
    const bool ext_dot = has_device_ext(c.phys, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    bool enable_dot = false;
    if (core13 || ext_dot) {
        VkPhysicalDeviceShaderIntegerDotProductFeatures df{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        VkPhysicalDeviceShaderIntegerDotProductProperties dp{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &df;
        vkGetPhysicalDeviceFeatures2(c.phys, &f2);
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &dp;
        vkGetPhysicalDeviceProperties2(c.phys, &p2);
        c.dot_feature             = df.shaderIntegerDotProduct == VK_TRUE;
        c.dot_packed_signed_fast  = dp.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
        // with the feature on, the packed signed form is supported; the property is only speed
        enable_dot = c.dot_feature && !force_fallback;
        c.dot_via_core13 = core13;
    }

    uint32_t nf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nf, nullptr);
    std::vector<VkQueueFamilyProperties> fams(nf);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nf, fams.data());
    int compute_fam = -1;
    for (uint32_t i = 0; i < nf; ++i) {
        const bool has_c = (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool has_g = (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (has_c && !has_g) {
            compute_fam = (int) i;
            break;
        }
        if (compute_fam < 0 && has_c) {
            compute_fam = (int) i;
        }
    }
    if (compute_fam < 0) {
        die_msg("no compute queue family");
    }
    c.fam = (uint32_t) compute_fam;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.shaderIntegerDotProduct = VK_TRUE;
    VkPhysicalDeviceShaderIntegerDotProductFeatures fdk{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    fdk.shaderIntegerDotProduct = VK_TRUE;

    // A 1.3 struct needs a 1.3 instance; otherwise the feature needs the KHR extension.
    const bool use_core13 = enable_dot && core13 && api >= VK_API_VERSION_1_3;
    if (enable_dot && !use_core13 && !ext_dot) {
        enable_dot = false;
    }

    std::vector<const char *> dev_exts;
    void * chain = &f12;
    if (use_core13) {
        f12.pNext = &f13;
    } else if (enable_dot) {
        fdk.pNext = &f12;
        chain = &fdk;
        dev_exts.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    }
    c.dot_enabled = enable_dot;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.fam;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext                   = chain;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = (uint32_t) dev_exts.size();
    dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
    VK_TRY(vkCreateDevice(c.phys, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.fam, 0, &c.queue);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = c.fam;
    VK_TRY(vkCreateCommandPool(c.dev, &cpci, nullptr, &c.pool));

    VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue  = 0;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &stci;
    VK_TRY(vkCreateSemaphore(c.dev, &sci, nullptr, &c.timeline));

    VkDescriptorSetLayoutBinding b[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 3;
    dslci.pBindings    = b;
    VK_TRY(vkCreateDescriptorSetLayout(c.dev, &dslci, nullptr, &c.dsl));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(Pc);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &c.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VK_TRY(vkCreatePipelineLayout(c.dev, &plci, nullptr, &c.pl));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    VK_TRY(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &c.dpool));
}

static uint32_t find_mem_type(const Ctx & c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.mem_props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0u &&
                (c.mem_props.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

static Buf make_buf(const Ctx & c, VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags want) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Buf b;
    VK_TRY(vkCreateBuffer(c.dev, &bci, nullptr, &b.buf));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(c.dev, b.buf, &req);
    const uint32_t type = find_mem_type(c, req.memoryTypeBits, want);
    if (type == UINT32_MAX) {
        std::printf("q4k_gemm: no memory type for %.1f MiB with flags 0x%x\n",
                (double) size / (1024.0 * 1024.0), (unsigned) want);
        std::exit(2);
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = type;
    const VkResult r = vkAllocateMemory(c.dev, &mai, nullptr, &b.mem);
    if (r != VK_SUCCESS) {
        std::printf("q4k_gemm: cannot allocate %.1f MiB, VkResult %d (free the GPU first)\n",
                (double) size / (1024.0 * 1024.0), (int) r);
        std::exit(2);
    }
    VK_TRY(vkBindBufferMemory(c.dev, b.buf, b.mem, 0));
    b.size = size;
    return b;
}

static void destroy_buf(const Ctx & c, Buf & b) {
    if (b.buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(c.dev, b.buf, nullptr);
    }
    if (b.mem != VK_NULL_HANDLE) {
        vkFreeMemory(c.dev, b.mem, nullptr);
    }
    b = Buf{};
}

static VkCommandBuffer alloc_cb(const Ctx & c) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = c.pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VK_TRY(vkAllocateCommandBuffers(c.dev, &cai, &cb));
    return cb;
}

static uint64_t submit_cb(Ctx & c, VkCommandBuffer cb) {
    const uint64_t value = ++c.seq;
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues    = &value;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext                = &ts;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &c.timeline;
    VK_TRY(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    return value;
}

static void wait_value(Ctx & c, uint64_t value) {
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores    = &c.timeline;
    wi.pValues        = &value;
    VK_TRY(vkWaitSemaphores(c.dev, &wi, UINT64_MAX));
}

// One-shot upload through a host-visible staging buffer; returns after the copy completes.
static void upload(Ctx & c, VkBuffer dst, const void * src, VkDeviceSize size) {
    Buf stage = make_buf(c, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void * p = nullptr;
    VK_TRY(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &p));
    std::memcpy(p, src, (size_t) size);
    vkUnmapMemory(c.dev, stage.mem);

    VkCommandBuffer cb = alloc_cb(c);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_TRY(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cb, stage.buf, dst, 1, &region);
    VK_TRY(vkEndCommandBuffer(cb));

    const uint64_t v = submit_cb(c, cb);
    wait_value(c, v);
    destroy_buf(c, stage);
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
}

static VkPipeline try_pipeline(Ctx & c, const char * spv_path) {
    std::ifstream f(spv_path, std::ios::binary | std::ios::ate);
    if (!f) {
        return VK_NULL_HANDLE;
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> code((size_t) size);
    f.read(code.data(), size);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode    = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c.dev, &smci, nullptr, &mod) != VK_SUCCESS) {
        std::printf("# q4k_gemm: %s is not a usable shader module\n", spv_path);
        return VK_NULL_HANDLE;
    }
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = c.pl;
    VkPipeline p = VK_NULL_HANDLE;
    const VkResult r = vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p);
    vkDestroyShaderModule(c.dev, mod, nullptr);
    if (r != VK_SUCCESS) {
        std::printf("# q4k_gemm: pipeline creation from %s failed, VkResult %d\n", spv_path, (int) r);
        return VK_NULL_HANDLE;
    }
    return p;
}

// ---------------------------------------------------------------- cpu reference

static uint32_t scale_of(const q4k_block & b, uint32_t j) {
    if (j < 4) {
        return b.scales[j] & 63u;
    }
    return (b.scales[j + 4] & 15u) | ((b.scales[j - 4] >> 6) << 4);
}

static uint32_t min_of(const q4k_block & b, uint32_t j) {
    if (j < 4) {
        return b.scales[j + 4] & 63u;
    }
    return (b.scales[j + 4] >> 4) | ((b.scales[j] >> 6) << 4);
}

// Direct double-precision dequant (canonical, matches dequantize_row_q4_K) plus the folded
// form the shader uses, so a check failure can be attributed either way.
struct RefPair {
    double direct;
    double folded;
    double abs_sum; // sum of |term| over the reduction, the error scale
};

static RefPair ref_element(const q4k_block * w, const q8_1_block * a, uint32_t row, uint32_t tok,
        uint32_t nk, uint32_t nk32) {
    const q4k_block *  wr = w + (size_t) row * nk;
    const q8_1_block * ar = a + (size_t) tok * nk32;
    double pos = 0.0;
    double abs_sum = 0.0;
    double min_direct = 0.0;
    double min_folded = 0.0;
    for (uint32_t sb = 0; sb < nk; ++sb) {
        const q4k_block & b = wr[sb];
        const double d    = f16_to_f32(b.d);
        const double dmin = f16_to_f32(b.dmin);
        for (uint32_t g = 0; g < Q4K_GROUPS; ++g) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t j  = 2 * g + half;
                const double   sc = (double) scale_of(b, j);
                const double   mn = (double) min_of(b, j);
                const q8_1_block & ab = ar[sb * BLOCKS_PER_SB + j];
                const double aq = f16_to_f32(ab.d);
                double s_block = 0.0;
                for (uint32_t e = 0; e < 32; ++e) {
                    const uint8_t byte = b.qs[32 * g + e];
                    const int q = half == 0 ? (byte & 0xF) : (byte >> 4);
                    const double term = (d * sc * aq) * ((double) q * (double) ab.qs[e]);
                    pos += term;
                    abs_sum += std::fabs(term);
                    s_block += (double) ab.qs[e];
                }
                min_direct += (dmin * mn) * (aq * s_block);
                min_folded += (dmin * mn) * (double) f16_to_f32(ab.s);
            }
        }
    }
    RefPair r;
    r.direct  = pos - min_direct;
    r.folded  = pos - min_folded;
    r.abs_sum = abs_sum;
    return r;
}

// ---------------------------------------------------------------- timing

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double median(std::vector<double> v) {
    if (v.empty()) {
        return -1.0;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------------------------------------------------------- args

struct Args {
    int         dev = -1;
    uint32_t    m = 2048;
    uint32_t    n = 512;
    uint32_t    k = 2048;
    uint32_t    reps = 32;
    uint32_t    warmup = 3;
    uint32_t    runs = 3;
    uint64_t    seed = 1;
    bool        check = false;
    bool        list = false;
    bool        fallback = false;
    std::string spv_dot = "q4k-gemm-dot.spv";
    std::string spv_fallback = "q4k-gemm-fallback.spv";
};

static void usage() {
    std::printf(
        "usage: q4k_gemm [options]\n"
        "  --dev N            physical device index (default: first discrete GPU)\n"
        "  --m N              tokens (default 2048, multiple of %d)\n"
        "  --n N              rows (default 512, multiple of %d)\n"
        "  --k N              reduction length (default 2048, multiple of %d)\n"
        "  --reps N           dispatches per timed submission (default 32)\n"
        "  --warmup N         warmup submissions (default 3)\n"
        "  --runs N           timed submissions, median reported (default 3)\n"
        "  --seed N           weight/activation seed (default 1)\n"
        "  --spv PATH         int8-dot shader (default q4k-gemm-dot.spv)\n"
        "  --spv-fallback PATH  portable shader (default q4k-gemm-fallback.spv)\n"
        "  --fallback         force the portable pipeline\n"
        "  --check            CPU cross-check of sampled outputs, one dispatch\n"
        "  --list             list devices and exit\n",
        Q4K_TILE_M, Q4K_TILE_N, Q4K_QK);
}

static uint32_t arg_u32(const char * name, const char * v) {
    char * end = nullptr;
    const unsigned long x = std::strtoul(v, &end, 10);
    if (end == v || *end != '\0' || x > 0xFFFFFFFFul) {
        std::printf("q4k_gemm: bad value for %s: %s\n", name, v);
        std::exit(1);
    }
    return (uint32_t) x;
}

static Args parse_args(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        const bool has_next = i + 1 < argc;
        if (f == "--help" || f == "-h") {
            usage();
            std::exit(0);
        } else if (f == "--check") {
            a.check = true;
        } else if (f == "--list") {
            a.list = true;
        } else if (f == "--fallback") {
            a.fallback = true;
        } else if (f == "--dev" && has_next) {
            a.dev = (int) arg_u32("--dev", argv[++i]);
        } else if (f == "--m" && has_next) {
            a.m = arg_u32("--m", argv[++i]);
        } else if (f == "--n" && has_next) {
            a.n = arg_u32("--n", argv[++i]);
        } else if (f == "--k" && has_next) {
            a.k = arg_u32("--k", argv[++i]);
        } else if (f == "--reps" && has_next) {
            a.reps = arg_u32("--reps", argv[++i]);
        } else if (f == "--warmup" && has_next) {
            a.warmup = arg_u32("--warmup", argv[++i]);
        } else if (f == "--runs" && has_next) {
            a.runs = arg_u32("--runs", argv[++i]);
        } else if (f == "--seed" && has_next) {
            a.seed = (uint64_t) arg_u32("--seed", argv[++i]);
        } else if (f == "--spv" && has_next) {
            a.spv_dot = argv[++i];
        } else if (f == "--spv-fallback" && has_next) {
            a.spv_fallback = argv[++i];
        } else {
            std::printf("q4k_gemm: unknown or incomplete option %s\n", f.c_str());
            usage();
            std::exit(1);
        }
    }
    return a;
}

// ---------------------------------------------------------------- main

static void shutdown(Ctx & c, Buf * bufs, uint32_t nbuf, VkCommandBuffer cb) {
    vkDeviceWaitIdle(c.dev);
    if (cb != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
    }
    for (uint32_t i = 0; i < nbuf; ++i) {
        destroy_buf(c, bufs[i]);
    }
    if (c.pipe != VK_NULL_HANDLE) {
        vkDestroyPipeline(c.dev, c.pipe, nullptr);
    }
    vkDestroyDescriptorPool(c.dev, c.dpool, nullptr);
    vkDestroyPipelineLayout(c.dev, c.pl, nullptr);
    vkDestroyDescriptorSetLayout(c.dev, c.dsl, nullptr);
    vkDestroySemaphore(c.dev, c.timeline, nullptr);
    vkDestroyCommandPool(c.dev, c.pool, nullptr);
    vkDestroyDevice(c.dev, nullptr);
    vkDestroyInstance(c.inst, nullptr);
}

int main(int argc, char ** argv) {
    const Args args = parse_args(argc, argv);
    if (args.reps == 0 || args.runs == 0) {
        std::printf("q4k_gemm: --reps and --runs must be positive\n");
        return 1;
    }
    if (args.n % Q4K_TILE_N != 0 || args.m % Q4K_TILE_M != 0 || args.k % QK_K != 0) {
        std::printf("q4k_gemm: need n%%%d==0, m%%%d==0, k%%%d==0 (got n=%u m=%u k=%u)\n",
                Q4K_TILE_N, Q4K_TILE_M, QK_K, args.n, args.m, args.k);
        return 1;
    }

    Ctx c;
    init_ctx(c, args.dev, args.fallback, args.list);
    std::printf("# q4k_gemm device=%s api=%s driver=0x%x\n", c.props.deviceName,
            version_str(c.props.apiVersion).c_str(), c.props.driverVersion);
    std::printf("# vulkan shaderIntegerDotProduct=%d packedSignedAccelerated=%d core13=%d"
            " enabled=%d\n", (int) c.dot_feature, (int) c.dot_packed_signed_fast,
            (int) c.dot_via_core13, (int) c.dot_enabled);


    const uint32_t nk    = args.k / QK_K;
    const uint32_t nk32  = args.k / QK8_1;
    const uint32_t n_gx  = args.n / Q4K_TILE_N;
    const uint32_t n_gy  = args.m / Q4K_TILE_M;

    std::vector<q4k_block>  w_blocks;
    std::vector<q8_1_block> a_blocks;
    Rng rng(args.seed);
    gen_weights(w_blocks, args.n, nk, rng);
    gen_acts(a_blocks, args.m, nk32, rng);
    const VkDeviceSize w_size = (VkDeviceSize) (w_blocks.size() * sizeof(q4k_block));
    const VkDeviceSize a_size = (VkDeviceSize) (a_blocks.size() * sizeof(q8_1_block));

    // Pick the pipeline: int8 dot when the device and shader agree, else the portable one.
    bool use_dot = c.dot_enabled;
    if (use_dot) {
        c.pipe = try_pipeline(c, args.spv_dot.c_str());
        if (c.pipe == VK_NULL_HANDLE) {
            std::printf("# q4k_gemm: no usable %s, falling back to %s\n", args.spv_dot.c_str(),
                    args.spv_fallback.c_str());
            use_dot = false;
        }
    }
    if (!use_dot) {
        c.pipe = try_pipeline(c, args.spv_fallback.c_str());
        if (c.pipe == VK_NULL_HANDLE) {
            std::printf("q4k_gemm: cannot create a pipeline, build the shaders first (build.sh)\n");
            return 2;
        }
    }
    std::printf("# pipeline=%s shader=%s\n", use_dot ? "dot" : "fallback",
            use_dot ? args.spv_dot.c_str() : args.spv_fallback.c_str());

    const VkDeviceSize o_size = (VkDeviceSize) args.n * args.m * sizeof(float);

    const VkMemoryPropertyFlags out_flags = args.check
            ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    Buf w_buf = make_buf(c, w_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf a_buf = make_buf(c, a_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf o_buf = make_buf(c, o_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out_flags);
    upload(c, w_buf.buf, w_blocks.data(), w_size);
    upload(c, a_buf.buf, a_blocks.data(), a_size);

    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool     = c.dpool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts        = &c.dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_TRY(vkAllocateDescriptorSets(c.dev, &dsa, &set));
    const VkDescriptorBufferInfo bi[3] = {{w_buf.buf, 0, VK_WHOLE_SIZE},
                                          {a_buf.buf, 0, VK_WHOLE_SIZE},
                                          {o_buf.buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        wds[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet          = set;
        wds[i].dstBinding      = i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(c.dev, 3, wds, 0, nullptr);

    Pc pc{args.n, args.k, args.m};

    std::printf("# shape m=%u n=%u k=%u tile=%dx%d wg=%d nk=%u\n", args.m, args.n, args.k,
            Q4K_TILE_N, Q4K_TILE_M, Q4K_WG_INVOCATIONS, nk);
    std::printf("# bytes weights=%llu activations=%llu out=%llu grid=%ux%u\n",
            (unsigned long long) w_size, (unsigned long long) a_size,
            (unsigned long long) o_size, n_gx, n_gy);

    if (args.check) {
        VkCommandBuffer cb = alloc_cb(c);
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_TRY(vkBeginCommandBuffer(cb, &cbbi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Pc), &pc);
        vkCmdDispatch(cb, n_gx, n_gy, 1);
        VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &bar, 0, nullptr, 0, nullptr);
        VK_TRY(vkEndCommandBuffer(cb));
        const uint64_t v = submit_cb(c, cb);
        wait_value(c, v);

        void * p = nullptr;
        VK_TRY(vkMapMemory(c.dev, o_buf.mem, 0, VK_WHOLE_SIZE, 0, &p));
        const float * out = (const float *) p;
        const q4k_block *  w = w_blocks.data();
        const q8_1_block * a = a_blocks.data();

        std::vector<uint32_t> rows;
        std::vector<uint32_t> toks;
        const uint32_t row_hint[] = {0, 1, 2, 3, 7, 15, 16, 31, 32, 33, 47, 62, 63, 64, 65,
                127, 128, 129, 255, 256, 257, args.n / 2, args.n - 2, args.n - 1};
        const uint32_t tok_hint[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 23, 24, 31, 32, 33, 63, 64,
                127, 128, 129, 1023, 1024, args.m / 2, args.m - 2, args.m - 1};
        for (uint32_t x : row_hint) {
            if (x < args.n && std::find(rows.begin(), rows.end(), x) == rows.end()) {
                rows.push_back(x);
            }
        }
        for (uint32_t x : tok_hint) {
            if (x < args.m && std::find(toks.begin(), toks.end(), x) == toks.end()) {
                toks.push_back(x);
            }
        }

        double max_abs = 0.0;
        double max_rel = 0.0;
        double max_abs_folded = 0.0;
        double max_rel_folded = 0.0;
        uint32_t samples = 0;
        uint32_t worst_row = 0;
        uint32_t worst_tok = 0;
        for (uint32_t row : rows) {
            for (uint32_t tok : toks) {
                const float got = out[(size_t) row * args.m + tok];
                const RefPair ref = ref_element(w, a, row, tok, nk, nk32);
                const double diff = std::fabs((double) got - ref.direct);
                const double denom = std::max(ref.abs_sum, 1.0);
                if (diff > max_abs) {
                    max_abs = diff;
                    worst_row = row;
                    worst_tok = tok;
                }
                max_rel = std::max(max_rel, diff / denom);
                const double difff = std::fabs((double) got - ref.folded);
                max_abs_folded = std::max(max_abs_folded, difff);
                max_rel_folded = std::max(max_rel_folded, difff / denom);
                samples++;
            }
        }
        vkUnmapMemory(c.dev, o_buf.mem);

        std::printf("# check rows=%zu tokens=%zu worst=(row=%u,tok=%u)\n", rows.size(), toks.size(),
                worst_row, worst_tok);
        std::printf("# folded form (shader algebra, fp16 s): max_abs=%.3e max_rel_l1=%.3e\n",
                max_abs_folded, max_rel_folded);
        std::printf("check,pipeline,samples,max_abs,max_rel_l1,tol,result\n");
        std::printf("check,%s,%u,%.3e,%.3e,%.1e,%s\n", use_dot ? "dot" : "fallback", samples,
                max_abs, max_rel, TOL_REL, max_rel <= TOL_REL ? "pass" : "fail");
        Buf bufs[3] = {w_buf, a_buf, o_buf};
        shutdown(c, bufs, 3, cb);
        return max_rel <= TOL_REL ? 0 : 1;
    }

    // Timed path: one command buffer with `reps` dispatches, submitted `runs` times.
    // No ONE_TIME_SUBMIT: the buffer is resubmitted after each timeline wait (not pending).
    VkCommandBuffer cb = alloc_cb(c);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = 0;
    VK_TRY(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Pc), &pc);
    for (uint32_t r = 0; r < args.reps; ++r) {
        vkCmdDispatch(cb, n_gx, n_gy, 1);
    }
    VK_TRY(vkEndCommandBuffer(cb));

    for (uint32_t i = 0; i < args.warmup; ++i) {
        const uint64_t v = submit_cb(c, cb);
        wait_value(c, v);
    }

    std::vector<double> per_dispatch;
    for (uint32_t run = 0; run < args.runs; ++run) {
        const Clock::time_point t0 = Clock::now();
        const uint64_t v = submit_cb(c, cb);
        wait_value(c, v);
        const Clock::time_point t1 = Clock::now();
        per_dispatch.push_back(ms_since(t0, t1) / (double) args.reps);
    }
    const double ms = median(per_dispatch);

    const double flops = 2.0 * (double) args.m * (double) args.n * (double) args.k;
    const double tflops = ms > 0.0 ? flops / (ms * 1e9) : 0.0;
    const double w_gbs = ms > 0.0 ? (double) w_size / (ms * 1e6) : 0.0;

    std::printf("# runs=%u per_dispatch_ms=", args.runs);
    for (size_t i = 0; i < per_dispatch.size(); ++i) {
        std::printf("%s%.4f", i == 0 ? "" : "/", per_dispatch[i]);
    }
    std::printf("\n");
    std::printf("# flops=%.4e per dispatch (2 per MAC), MACs=%.4e\n", flops,
            flops * 0.5);
    std::printf("shape,reps,runs,ms,tflops,w_gbs\n");
    std::printf("%ux%ux%u,%u,%u,%.4f,%.3f,%.2f\n", args.m, args.n, args.k, args.reps,
            args.runs, ms, tflops, w_gbs);
    std::fflush(stdout);

    Buf bufs[3] = {w_buf, a_buf, o_buf};
    shutdown(c, bufs, 3, cb);
    return 0;
}
