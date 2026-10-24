// vkgate.cpp - Gate 1 (per-layer host/GPU boundary cost) and Gate 2 (transfer vs compute overlap).
// Standalone Vulkan probe. It does not link llama.cpp/ggml; it reproduces the two costs that
// decide whether a host-driven MoE executor can beat the current scheduler on this machine.
//
// Build (MinGW, Vulkan SDK):
//   glslc -O gemv.comp -o gemv.spv
//   glslc -O ids.comp  -o ids.spv
//   g++ -O2 -std=c++17 vkgate.cpp -I"%VULKAN_SDK%\Include" -L"%VULKAN_SDK%\Lib" -lvulkan-1 -o vkgate.exe
//
// Usage: vkgate.exe info | boundary [iters] | overlap [transfer_mib] [target_ms]

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                                       \
    do {                                                                                  \
        VkResult r_ = (x);                                                                \
        if (r_ != VK_SUCCESS) {                                                           \
            std::printf("VK error %d at %s:%d\n", (int) r_, __FILE__, __LINE__);          \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

static constexpr VkDeviceSize MIB = 1024ull * 1024ull;

static std::vector<char> read_file(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::printf("cannot open %s\n", path);
        std::exit(1);
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data((size_t) size);
    f.read(data.data(), size);
    return data;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Buf {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
    void *         map = nullptr;
};

struct App {
    VkInstance               inst = VK_NULL_HANDLE;
    VkPhysicalDevice         phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties       props{};
    VkPhysicalDeviceMemoryProperties memprops{};
    VkDevice                 dev = VK_NULL_HANDLE;

    uint32_t fam_compute  = 0;
    uint32_t fam_transfer = 0;
    bool     dedicated_transfer = false;
    uint32_t compute_ts_bits = 0;
    std::vector<uint32_t> share_fams;
    VkQueue  qc = VK_NULL_HANDLE;
    VkQueue  qt = VK_NULL_HANDLE;

    VkCommandPool pool_c = VK_NULL_HANDLE;
    VkCommandPool pool_t = VK_NULL_HANDLE;

    // separate timeline semaphores per queue: a single one cannot be waited per queue safely
    VkSemaphore sem_c = VK_NULL_HANDLE;
    VkSemaphore sem_t = VK_NULL_HANDLE;
    uint64_t    val_c = 0;
    uint64_t    val_t = 0;

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout      pl  = VK_NULL_HANDLE;
    VkPipeline            pipe_gemv = VK_NULL_HANDLE;
    VkPipeline            pipe_gemv4 = VK_NULL_HANDLE;
    VkPipeline            pipe_ids  = VK_NULL_HANDLE;
    VkDescriptorPool      dpool = VK_NULL_HANDLE;

    VkQueryPool qpool = VK_NULL_HANDLE;

    Buf layer_w, big_w, out_buf, ids_dev, ids_host, src_host, dst_dev;
};

static uint32_t mem_type(App & a, uint32_t bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < a.memprops.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (a.memprops.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    std::printf("no memory type for bits=0x%x props=0x%x\n", bits, props);
    std::exit(1);
}

static Buf make_buf(App & a, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = size;
    bci.usage       = usage;
    // buffers are used from both the compute and the transfer family: share concurrently
    bci.sharingMode           = VK_SHARING_MODE_CONCURRENT;
    bci.queueFamilyIndexCount = (uint32_t) a.share_fams.size();
    bci.pQueueFamilyIndices   = a.share_fams.data();
    VK_CHECK(vkCreateBuffer(a.dev, &bci, nullptr, &b.buf));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(a.dev, b.buf, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = mem_type(a, req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(a.dev, &mai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(a.dev, b.buf, b.mem, 0));
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK_CHECK(vkMapMemory(a.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    }
    return b;
}

static VkPipeline make_pipeline(App & a, const char * spv_path) {
    const std::vector<char> code = read_file(spv_path);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode    = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(a.dev, &smci, nullptr, &mod));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = a.pl;
    VkPipeline p = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(a.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
    vkDestroyShaderModule(a.dev, mod, nullptr);
    return p;
}

static VkCommandBuffer alloc_cb(App & a, VkCommandPool pool) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(a.dev, &cai, &cb));
    return cb;
}

static void begin_cb(VkCommandBuffer cb) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
}

static void submit_tl(App & a, VkQueue q, VkSemaphore sem, uint64_t * counter,
        VkCommandBuffer cb, uint64_t * value) {
    *value = ++(*counter);
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues    = value;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext                = &ts;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &sem;
    VK_CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
}

static void wait_tl(App & a, VkSemaphore sem, uint64_t value) {
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores    = &sem;
    wi.pValues        = &value;
    VK_CHECK(vkWaitSemaphores(a.dev, &wi, UINT64_MAX));
}

static void submit_fence(App & a, VkQueue q, VkCommandBuffer cb) {
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(a.dev, &fci, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VK_CHECK(vkQueueSubmit(q, 1, &si, fence));
    VK_CHECK(vkWaitForFences(a.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(a.dev, fence, nullptr);
}

static void init(App & a) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &a.inst));

    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(a.inst, &n, nullptr));
    std::vector<VkPhysicalDevice> devs(n);
    VK_CHECK(vkEnumeratePhysicalDevices(a.inst, &n, devs.data()));

    std::printf("devices:\n");
    int pick = -1;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        std::printf("  [%u] %s (type=%d api=%u.%u.%u)\n", i, p.deviceName, (int) p.deviceType,
                VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion));
        if (pick < 0 && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            pick = (int) i;
        }
    }
    if (pick < 0) {
        pick = 0;
    }
    a.phys = devs[pick];
    vkGetPhysicalDeviceProperties(a.phys, &a.props);
    vkGetPhysicalDeviceMemoryProperties(a.phys, &a.memprops);
    std::printf("picked: %s\n", a.props.deviceName);

    uint32_t nf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(a.phys, &nf, nullptr);
    std::vector<VkQueueFamilyProperties> fams(nf);
    vkGetPhysicalDeviceQueueFamilyProperties(a.phys, &nf, fams.data());
    std::printf("queue families:\n");
    int compute_fam = -1, transfer_fam = -1, transfer_only = -1;
    for (uint32_t i = 0; i < nf; ++i) {
        std::printf("  family %u: flags=0x%x count=%u timestampValidBits=%u\n", i, fams[i].queueFlags,
                fams[i].queueCount, fams[i].timestampValidBits);
        const bool has_c = (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool has_t = (fams[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
        const bool has_g = (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        // mirror ggml-vulkan: prefer a compute family that is not the graphics family
        if (has_c && !has_g) {
            compute_fam = (int) i;
        } else if (compute_fam < 0 && has_c) {
            compute_fam = (int) i;
        }
        if (transfer_only < 0 && has_t && !has_c && !has_g) {
            transfer_only = (int) i;
        }
        if (transfer_fam < 0 && has_t && (int) i != compute_fam) {
            transfer_fam = (int) i;
        }
    }
    if (compute_fam < 0) {
        std::printf("no compute queue family\n");
        std::exit(1);
    }
    a.fam_compute     = (uint32_t) compute_fam;
    a.compute_ts_bits = fams[compute_fam].timestampValidBits;
    if (transfer_only >= 0) {
        a.fam_transfer       = (uint32_t) transfer_only;
        a.dedicated_transfer = true;
    } else if (transfer_fam >= 0) {
        a.fam_transfer       = (uint32_t) transfer_fam;
        a.dedicated_transfer = false;
    } else {
        a.fam_transfer       = a.fam_compute;
        a.dedicated_transfer = false;
    }
    a.share_fams.push_back(a.fam_compute);
    if (a.fam_transfer != a.fam_compute) {
        a.share_fams.push_back(a.fam_transfer);
    }
    std::printf("compute family=%u transfer family=%u dedicated_transfer=%d\n",
            a.fam_compute, a.fam_transfer, (int) a.dedicated_transfer);

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(a.phys, &f2);
    std::printf("timelineSemaphore=%d\n", (int) f12.timelineSemaphore);
    if (!f12.timelineSemaphore) {
        std::printf("timeline semaphores are required for this probe\n");
        std::exit(1);
    }

    float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    {
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = a.fam_compute;
        q.queueCount       = 1;
        q.pQueuePriorities = &prio;
        qcis.push_back(q);
    }
    if (a.fam_transfer != a.fam_compute) {
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = a.fam_transfer;
        q.queueCount       = 1;
        q.pQueuePriorities = &prio;
        qcis.push_back(q);
    }

    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enable12.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext                = &enable12;
    dci.queueCreateInfoCount = (uint32_t) qcis.size();
    dci.pQueueCreateInfos    = qcis.data();
    VK_CHECK(vkCreateDevice(a.phys, &dci, nullptr, &a.dev));
    vkGetDeviceQueue(a.dev, a.fam_compute, 0, &a.qc);
    vkGetDeviceQueue(a.dev, a.fam_transfer, 0, &a.qt);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = a.fam_compute;
    VK_CHECK(vkCreateCommandPool(a.dev, &cpci, nullptr, &a.pool_c));
    cpci.queueFamilyIndex = a.fam_transfer;
    VK_CHECK(vkCreateCommandPool(a.dev, &cpci, nullptr, &a.pool_t));

    VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue  = 0;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &stci;
    VK_CHECK(vkCreateSemaphore(a.dev, &sci, nullptr, &a.sem_c));
    VK_CHECK(vkCreateSemaphore(a.dev, &sci, nullptr, &a.sem_t));

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
    VK_CHECK(vkCreateDescriptorSetLayout(a.dev, &dslci, nullptr, &a.dsl));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = 8;
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &a.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(a.dev, &plci, nullptr, &a.pl));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets       = 1024;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    VK_CHECK(vkCreateDescriptorPool(a.dev, &dpci, nullptr, &a.dpool));

    a.pipe_gemv = make_pipeline(a, "gemv.spv");
    a.pipe_ids  = make_pipeline(a, "ids.spv");
    a.pipe_gemv4 = make_pipeline(a, "gemv4.spv");

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;
    VK_CHECK(vkCreateQueryPool(a.dev, &qpci, nullptr, &a.qpool));
}

static void alloc_buffers(App & a) {
    a.layer_w = make_buf(a, 14 * MIB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // 256 MiB does not fit in the 32 MB Infinity Cache + L2, so the compute arm streams from DRAM
    a.big_w   = make_buf(a, 256 * MIB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    a.out_buf  = make_buf(a, 16 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    a.ids_dev  = make_buf(a, 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    a.ids_host = make_buf(a, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    a.src_host = make_buf(a, 64 * MIB, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    a.dst_dev  = make_buf(a, 16 * MIB, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::memset(a.src_host.map, 0x5a, (size_t) a.src_host.size);
}

static VkDescriptorSet make_set(App & a, VkBuffer src) {
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = a.dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &a.dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(a.dev, &ai, &set));
    VkDescriptorBufferInfo bi0{src, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo bi1{a.out_buf.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2]{};
    w[0].sType = w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = w[1].dstSet = set;
    w[0].dstBinding = 0;
    w[1].dstBinding = 1;
    w[0].descriptorCount = w[1].descriptorCount = 1;
    w[0].descriptorType = w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].pBufferInfo = &bi0;
    w[1].pBufferInfo = &bi1;
    vkUpdateDescriptorSets(a.dev, 2, w, 0, nullptr);
    return set;
}

struct Work {
    uint32_t n      = 1;
    uint32_t reps   = 1;
    uint32_t groups = 1024; // fixed grid; each thread strided-reads many elements
};

static Work layer_work() {
    Work w;
    w.n    = (uint32_t) (14 * MIB / 4); // one layer's expert bytes read
    w.reps = 4;
    return w;
}

static Work big_work(uint32_t reps) {
    Work w;
    w.n    = (uint32_t) (256 * MIB / 4);
    w.reps = reps;
    return w;
}

// gemv dispatch + ids dispatch + ids copy to host. Timestamps are optional.
static void record_layer(App & a, VkCommandBuffer cb, VkDescriptorSet set, Work w, bool timestamps) {
    if (timestamps) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, a.qpool, 0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, a.pipe_gemv);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, a.pl, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, a.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &w);
    vkCmdDispatch(cb, w.groups, 1, 1);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, a.pipe_ids);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, a.pl, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cb, 1, 1, 1);

    VkBufferMemoryBarrier bar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bar.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.buffer = a.ids_dev.buf;
    bar.offset = 0;
    bar.size   = 128;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr, 1, &bar, 0, nullptr);
    VkBufferCopy cpy{0, 0, 128};
    vkCmdCopyBuffer(cb, a.ids_dev.buf, a.ids_host.buf, 1, &cpy);
    if (timestamps) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, a.qpool, 1);
    }
}

static double gpu_work_ms(App & a, Work w, VkBuffer src) {
    if (a.compute_ts_bits == 0) {
        return -1.0;
    }
    VkDescriptorSet set = make_set(a, src);
    VkCommandBuffer cb = alloc_cb(a, a.pool_c);
    vkResetQueryPool(a.dev, a.qpool, 0, 2);
    begin_cb(cb);
    record_layer(a, cb, set, w, true);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_fence(a, a.qc, cb);
    uint64_t ts[2]{};
    VK_CHECK(vkGetQueryPoolResults(a.dev, a.qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    const double period = a.props.limits.timestampPeriod;
    return (double) (ts[1] - ts[0]) * period / 1e6;
}

static void cmd_boundary(App & a, int iters) {
    const Work w = layer_work();
    const double work = gpu_work_ms(a, w, a.layer_w.buf);
    if (work > 0) {
        std::printf("recorded GPU work per layer: %.3f ms (gemv n=%u reps=%u + ids + 128 B copy)\n",
                work, w.n, w.reps);
    } else {
        std::printf("timestamps unavailable on the compute family; reporting total only\n");
    }

    struct Variant {
        const char * name;
        int          kind; // 0 per-iter alloc+record+fence, 1 prerecorded+fence,
                           // 2 prerecorded+timeline, 3 pipelined depth2+timeline
    } variants[] = {
        {"A alloc+record+fence (scheduler-like)", 0},
        {"B prerecorded+fence",                   1},
        {"C prerecorded+timeline",                2},
        {"D pipelined depth2+timeline",           3},
    };

    for (const Variant & v : variants) {
        for (int i = 0; i < 8; ++i) {
            VkDescriptorSet set = make_set(a, a.layer_w.buf);
            VkCommandBuffer cb  = alloc_cb(a, a.pool_c);
            begin_cb(cb);
            record_layer(a, cb, set, w, false);
            VK_CHECK(vkEndCommandBuffer(cb));
            submit_fence(a, a.qc, cb);
            if (v.kind == 0) {
                vkFreeCommandBuffers(a.dev, a.pool_c, 1, &cb);
                vkFreeDescriptorSets(a.dev, a.dpool, 1, &set);
            }
        }
        vkQueueWaitIdle(a.qc);
        vkResetCommandPool(a.dev, a.pool_c, 0);
        vkResetDescriptorPool(a.dev, a.dpool, 0);

        VkDescriptorSet set1 = VK_NULL_HANDLE, set2 = VK_NULL_HANDLE;
        VkCommandBuffer cb1 = VK_NULL_HANDLE, cb2 = VK_NULL_HANDLE;
        if (v.kind != 0) {
            set1 = make_set(a, a.layer_w.buf);
            cb1  = alloc_cb(a, a.pool_c);
            begin_cb(cb1);
            record_layer(a, cb1, set1, w, false);
            VK_CHECK(vkEndCommandBuffer(cb1));
            if (v.kind == 3) {
                set2 = make_set(a, a.layer_w.buf);
                cb2  = alloc_cb(a, a.pool_c);
                begin_cb(cb2);
                record_layer(a, cb2, set2, w, false);
                VK_CHECK(vkEndCommandBuffer(cb2));
            }
        }

        volatile uint32_t sink = 0;
        uint64_t last = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (v.kind == 0) {
                VkDescriptorSet set = make_set(a, a.layer_w.buf);
                VkCommandBuffer cb  = alloc_cb(a, a.pool_c);
                begin_cb(cb);
                record_layer(a, cb, set, w, false);
                VK_CHECK(vkEndCommandBuffer(cb));
                submit_fence(a, a.qc, cb);
                sink += ((const uint32_t *) a.ids_host.map)[0];
                vkFreeCommandBuffers(a.dev, a.pool_c, 1, &cb);
                vkFreeDescriptorSets(a.dev, a.dpool, 1, &set);
            } else if (v.kind == 1) {
                submit_fence(a, a.qc, cb1);
                sink += ((const uint32_t *) a.ids_host.map)[0];
            } else if (v.kind == 2) {
                uint64_t val = 0;
                submit_tl(a, a.qc, a.sem_c, &a.val_c, cb1, &val);
                wait_tl(a, a.sem_c, val);
                sink += ((const uint32_t *) a.ids_host.map)[0];
            } else {
                uint64_t val = 0;
                submit_tl(a, a.qc, a.sem_c, &a.val_c, (i & 1) ? cb2 : cb1, &val);
                if (last != 0) {
                    wait_tl(a, a.sem_c, last);
                }
                last = val;
                sink += ((const uint32_t *) a.ids_host.map)[0];
            }
        }
        if (v.kind == 3 && last != 0) {
            wait_tl(a, a.sem_c, last);
        }
        const double total = ms_since(t0);
        const double per   = total / iters;
        if (work > 0) {
            std::printf("%-40s %8.3f ms/layer  overhead=%+.3f ms/layer\n", v.name, per, per - work);
        } else {
            std::printf("%-40s %8.3f ms/layer\n", v.name, per);
        }
        (void) sink;
        vkQueueWaitIdle(a.qc);
        vkResetCommandPool(a.dev, a.pool_c, 0);
        vkResetDescriptorPool(a.dev, a.dpool, 0);
    }
}

static void record_compute(App & a, VkCommandBuffer cb, VkDescriptorSet set, Work w) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, a.pipe_gemv);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, a.pl, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, a.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &w);
    vkCmdDispatch(cb, w.groups, 1, 1);
}

static void record_transfer(App & a, VkCommandBuffer cb, uint32_t mib) {
    const VkDeviceSize chunk = 6 * MIB + 700 * 1024; // 6.7 MiB, one layer of misses
    const uint32_t     ncopy = (uint32_t) ((VkDeviceSize) mib * MIB / chunk);
    for (uint32_t i = 0; i < ncopy; ++i) {
        VkBufferCopy c{0, (i & 1) ? 8 * MIB : 0, chunk};
        vkCmdCopyBuffer(cb, a.src_host.buf, a.dst_dev.buf, 1, &c);
    }
}

static double time_compute(App & a, uint32_t reps) {
    const Work w = big_work(reps);
    VkDescriptorSet set = make_set(a, a.big_w.buf);
    VkCommandBuffer cb  = alloc_cb(a, a.pool_c);
    begin_cb(cb);
    record_compute(a, cb, set, w);
    VK_CHECK(vkEndCommandBuffer(cb));
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t val = 0;
    submit_tl(a, a.qc, a.sem_c, &a.val_c, cb, &val);
    wait_tl(a, a.sem_c, val);
    const double ms = ms_since(t0);
    vkResetCommandPool(a.dev, a.pool_c, 0);
    vkResetDescriptorPool(a.dev, a.dpool, 0);
    return ms;
}

static double time_transfer(App & a, uint32_t mib) {
    VkCommandBuffer cb = alloc_cb(a, a.pool_t);
    begin_cb(cb);
    record_transfer(a, cb, mib);
    VK_CHECK(vkEndCommandBuffer(cb));
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t val = 0;
    submit_tl(a, a.qt, a.sem_t, &a.val_t, cb, &val);
    wait_tl(a, a.sem_t, val);
    const double ms = ms_since(t0);
    vkResetCommandPool(a.dev, a.pool_t, 0);
    return ms;
}

static void cmd_overlap(App & a, uint32_t transfer_mib, double target_ms) {
    // calibrate the compute arms on the 256 MiB buffer (DRAM-bound, not cache-resident)
    const double calib = time_compute(a, 4);
    uint32_t reps = (uint32_t) std::max(1.0, 4.0 * target_ms / std::max(0.001, calib));

    const double t_c = time_compute(a, reps);
    const double t_t = time_transfer(a, transfer_mib);
    const double gb_c = (double) reps * 256.0 / 1024.0;
    const double gb_t = (double) transfer_mib / 1024.0;
    std::printf("compute arm: reps=%u (%.1f GiB read) -> %.1f ms (%.1f GB/s)\n",
            reps, gb_c, t_c, gb_c * 1000.0 / t_c);
    std::printf("transfer arm: %u MiB (%.1f GiB) -> %.1f ms (%.1f GB/s)\n",
            transfer_mib, gb_t, t_t, gb_t * 1000.0 / t_t);

    const Work w = big_work(reps);
    {
        VkDescriptorSet set = make_set(a, a.big_w.buf);
        VkCommandBuffer cb_c = alloc_cb(a, a.pool_c);
        begin_cb(cb_c);
        record_compute(a, cb_c, set, w);
        VK_CHECK(vkEndCommandBuffer(cb_c));
        VkCommandBuffer cb_t = alloc_cb(a, a.pool_c); // compute-family pool: submitted on qc
        begin_cb(cb_t);
        record_transfer(a, cb_t, transfer_mib);
        VK_CHECK(vkEndCommandBuffer(cb_t));

        const auto t0 = std::chrono::steady_clock::now();
        uint64_t v1 = 0, v2 = 0;
        submit_tl(a, a.qc, a.sem_c, &a.val_c, cb_t, &v1);
        wait_tl(a, a.sem_c, v1);
        submit_tl(a, a.qc, a.sem_c, &a.val_c, cb_c, &v2);
        wait_tl(a, a.sem_c, v2);
        std::printf("same-queue serial: %.1f ms (t_c+t_t=%.1f)\n", ms_since(t0), t_c + t_t);
        vkResetCommandPool(a.dev, a.pool_c, 0);
        vkResetDescriptorPool(a.dev, a.dpool, 0);
    }

    VkDescriptorSet set = make_set(a, a.big_w.buf);
    VkCommandBuffer cb_c = alloc_cb(a, a.pool_c);
    begin_cb(cb_c);
    record_compute(a, cb_c, set, w);
    VK_CHECK(vkEndCommandBuffer(cb_c));
    VkCommandBuffer cb_t = alloc_cb(a, a.pool_t);
    begin_cb(cb_t);
    record_transfer(a, cb_t, transfer_mib);
    VK_CHECK(vkEndCommandBuffer(cb_t));

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t v_t = 0, v_c = 0;
    submit_tl(a, a.qt, a.sem_t, &a.val_t, cb_t, &v_t);
    submit_tl(a, a.qc, a.sem_c, &a.val_c, cb_c, &v_c);
    wait_tl(a, a.sem_c, v_c);
    wait_tl(a, a.sem_t, v_t);
    const double t_sum = ms_since(t0);

    const double serial = t_c + t_t;
    const double hidden = serial - t_sum;
    const double denom  = std::min(t_c, t_t);
    std::printf("two queues, both submitted before waiting: %.1f ms (t_c+t_t=%.1f)\n", t_sum, serial);
    std::printf("overlap hidden: %.1f ms = %.1f%% of min(t_c,t_t)=%.1f ms\n", hidden,
            100.0 * hidden / denom, denom);
    std::printf("VERDICT: %s\n", hidden > 0.5 * denom ? "transfer and compute DO overlap" :
            (hidden > 0.15 * denom ? "partial overlap" : "no measurable overlap"));
}

// vec4 variant of record_compute: n is counted in vec4 elements.
static void record_compute4(App & a, VkCommandBuffer cb, VkDescriptorSet set, Work w) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, a.pipe_gemv4);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, a.pl, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, a.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &w);
    vkCmdDispatch(cb, w.groups, 1, 1);
}

// Streams a large device buffer to measure achievable VRAM read bandwidth. Sizes above the
// 32 MB Infinity Cache keep the read off the cache, so the number is a DRAM read ceiling.
static void cmd_stream(App & a, uint32_t mib, uint32_t dispatches, bool vec4) {
    Buf stream_w = make_buf(a, (VkDeviceSize) mib * MIB,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Work w;
    w.n    = vec4 ? (uint32_t) ((VkDeviceSize) mib * MIB / 16) : (uint32_t) ((VkDeviceSize) mib * MIB / 4);
    w.reps = 1;
    VkDescriptorSet set = make_set(a, stream_w.buf);

    // one untimed pass so the first touch does not land in the measured window
    {
        VkCommandBuffer cb = alloc_cb(a, a.pool_c);
        begin_cb(cb);
        if (vec4) { record_compute4(a, cb, set, w); } else { record_compute(a, cb, set, w); }
        VK_CHECK(vkEndCommandBuffer(cb));
        uint64_t val = 0;
        submit_tl(a, a.qc, a.sem_c, &a.val_c, cb, &val);
        wait_tl(a, a.sem_c, val);
        vkResetCommandPool(a.dev, a.pool_c, 0);
    }

    VkCommandBuffer cb = alloc_cb(a, a.pool_c);
    begin_cb(cb);
    for (uint32_t i = 0; i < dispatches; ++i) {
        if (vec4) { record_compute4(a, cb, set, w); } else { record_compute(a, cb, set, w); }
    }
    VK_CHECK(vkEndCommandBuffer(cb));

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t val = 0;
    submit_tl(a, a.qc, a.sem_c, &a.val_c, cb, &val);
    wait_tl(a, a.sem_c, val);
    const double ms  = ms_since(t0);
    const double gb  = (double) mib * dispatches / 1024.0;
    std::printf("%u MiB x %u dispatches: %.2f GiB read in %.2f ms -> %.1f GB/s\n",
            mib, dispatches, gb, ms, gb / (ms / 1000.0));
    vkQueueWaitIdle(a.qc);
    vkDestroyBuffer(a.dev, stream_w.buf, nullptr);
    vkFreeMemory(a.dev, stream_w.mem, nullptr);
}

int main(int argc, char ** argv) {
    App a;
    init(a);
    alloc_buffers(a);

    const std::string mode = argc > 1 ? argv[1] : "info";
    if (mode == "boundary") {
        cmd_boundary(a, argc > 2 ? std::atoi(argv[2]) : 200);
    } else if (mode == "overlap") {
        cmd_overlap(a, argc > 2 ? (uint32_t) std::atoi(argv[2]) : 1400,
                argc > 3 ? std::atof(argv[3]) : 100.0);
    } else if (mode == "stream") {
        cmd_stream(a, argc > 2 ? (uint32_t) std::atoi(argv[2]) : 1024,
                argc > 3 ? (uint32_t) std::atoi(argv[3]) : 16, false);
    } else if (mode == "stream4") {
        cmd_stream(a, argc > 2 ? (uint32_t) std::atoi(argv[2]) : 1024,
                argc > 3 ? (uint32_t) std::atoi(argv[3]) : 16, true);
    } else {
        std::printf("usage: %s info | boundary [iters] | overlap [transfer_mib] [target_ms]"
                    " | stream [buffer_mib] [dispatches] | stream4 [buffer_mib] [dispatches]\n", argv[0]);
    }
    return 0;
}
