// Does a GPU-driven copy kernel beat the SDMA engine?
// Same buffers as h2d-bw.cpp, but the transfer is done by a compute kernel that
// reads host memory over PCIe instead of vkCmdCopyBuffer. Also runs kernel legs
// concurrently with SDMA legs to see whether the two paths share one budget.
// Build: see BUILD.txt.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <thread>
#include <vector>

#define VK_CHECK(expr)                                                       \
    do {                                                                      \
        VkResult r_ = (expr);                                                 \
        if (r_ != VK_SUCCESS && r_ != VK_INCOMPLETE) {                         \
            printf("FAIL %s -> %d (line %d)\n", #expr, (int) r_, __LINE__);    \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static const double GB = 1e9;

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint8_t * persist = nullptr;
};

static VkInstance g_inst;
static VkPhysicalDevice g_phys;
static VkDevice g_dev;
static VkQueue g_q0, g_q1;
static VkCommandPool g_pool0, g_pool1;
static uint32_t g_memTypeAll = 0xFFFFFFFFu;

static uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return 0xFFFFFFFFu;
}

static void createBuf(Buf & b, VkDeviceSize size, VkMemoryPropertyFlags want, const char * label) {
    b.size = size;
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(g_dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g_dev, b.buf, &req);
    uint32_t t = findMemoryType(req.memoryTypeBits, want);
    if (t == 0xFFFFFFFFu) { printf("no memory type for %s\n", label); std::exit(1); }
    VkMemoryDedicatedAllocateInfo ded{ VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    ded.buffer = b.buf;
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = t;
    ai.pNext = &ded;
    VK_CHECK(vkAllocateMemory(g_dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(g_dev, b.buf, b.mem, 0));
}

static void fillHost(Buf & b, bool keepMapped) {
    void * p = nullptr;
    VK_CHECK(vkMapMemory(g_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &p));
    std::memset(p, 0x5a, (size_t) b.size);
    if (keepMapped) b.persist = (uint8_t *) p;
    else vkUnmapMemory(g_dev, b.mem);
}

static VkPipeline g_pipe;
static VkPipelineLayout g_layout;
static VkDescriptorPool g_dpool;
static VkDescriptorSetLayout g_dsetLayout;

struct Desc {
    VkDescriptorSet set;
};

static void makePipeline(const char * spvPath) {
    std::ifstream f(spvPath, std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (code.empty()) { printf("no shader at %s\n", spvPath); std::exit(1); }

    VkDescriptorSetLayoutBinding b0{ 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutBinding b1{ 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo dli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dli.bindingCount = 2;
    VkDescriptorSetLayoutBinding binds[2] = { b0, b1 };
    dli.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(g_dev, &dli, nullptr, &g_dsetLayout));

    VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };
    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_dsetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(g_dev, &pli, nullptr, &g_layout));

    VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = code.size();
    smi.pCode = (const uint32_t *) code.data();
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(g_dev, &smi, nullptr, &mod));

    VkPipelineShaderStageCreateInfo stg{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stg.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stg.module = mod;
    stg.pName = "main";
    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage = stg;
    cpi.layout = g_layout;
    VK_CHECK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &g_pipe));
    vkDestroyShaderModule(g_dev, mod, nullptr);

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 };
    VkDescriptorPoolCreateInfo dpi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = 8;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(g_dev, &dpi, nullptr, &g_dpool));
}

static Desc makeDesc(const Buf & src, const Buf & dst) {
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = g_dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g_dsetLayout;
    Desc d;
    VK_CHECK(vkAllocateDescriptorSets(g_dev, &ai, &d.set));
    VkDescriptorBufferInfo si{ src.buf, 0, src.size }, di{ dst.buf, 0, dst.size };
    VkWriteDescriptorSet w[2]{
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, d.set, 0, 0, 1,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &si, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, d.set, 1, 0, 1,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &di, nullptr },
    };
    vkUpdateDescriptorSets(g_dev, 2, w, 0, nullptr);
    return d;
}

static void recordKernel(VkCommandBuffer cb, const Desc & d, VkDeviceSize bytes, int wg) {
    const uint32_t threads = (uint32_t) wg * 256u;
    const uint32_t N = (uint32_t) (bytes / 4);
    const uint32_t per = threads * 4u;
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_layout, 0, 1, &d.set, 0, nullptr);
    for (uint32_t base = 0; base + per <= N; base += per) {
        uint32_t pc[2] = { base, threads };
        vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
        vkCmdDispatch(cb, (uint32_t) wg, 1, 1);
    }
    VK_CHECK(vkEndCommandBuffer(cb));
}

static double timedSubmit(VkCommandBuffer cb, VkQueue q) {
    VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence f;
    VK_CHECK(vkCreateFence(g_dev, &fi, nullptr, &f));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    auto t0 = std::chrono::steady_clock::now();
    VK_CHECK(vkQueueSubmit(q, 1, &si, f));
    VK_CHECK(vkWaitForFences(g_dev, 1, &f, VK_TRUE, ~0ull));
    auto t1 = std::chrono::steady_clock::now();
    vkDestroyFence(g_dev, f, nullptr);
    return std::chrono::duration<double>(t1 - t0).count();
}

static double sdma(const Buf & src, const Buf & dst, VkDeviceSize bytes, VkQueue q, VkCommandPool pool) {
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = pool;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(g_dev, &cai, &cb));
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy c{ 0, 0, bytes };
    vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &c);
    VK_CHECK(vkEndCommandBuffer(cb));
    double sec = timedSubmit(cb, q);
    vkFreeCommandBuffers(g_dev, pool, 1, &cb);
    return (double) bytes / sec;
}

int main(int argc, char ** argv) {
    VkDeviceSize region = 512ull << 20;
    if (argc > 1) region = (VkDeviceSize) std::strtoull(argv[1], nullptr, 10);

    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &g_inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_inst, &n, nullptr);
    std::vector<VkPhysicalDevice> ps(n);
    vkEnumeratePhysicalDevices(g_inst, &n, ps.data());
    g_phys = ps[0];
    for (auto p : ps) {
        VkPhysicalDeviceProperties pp{};
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) g_phys = p;
    }
    VkPhysicalDeviceProperties pprops{};
    vkGetPhysicalDeviceProperties(g_phys, &pprops);
    printf("device %s  maxWgCount=%u %u %u  maxInvocations=%u\n", pprops.deviceName,
           pprops.limits.maxComputeWorkGroupCount[0], pprops.limits.maxComputeWorkGroupCount[1],
           pprops.limits.maxComputeWorkGroupCount[2], pprops.limits.maxComputeWorkGroupInvocations);

    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &n, fams.data());
    int gfx = -1, xferOnly = -1;
    for (uint32_t i = 0; i < n; i++) {
        bool isG = fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
        bool isT = fams[i].queueFlags & VK_QUEUE_TRANSFER_BIT;
        bool isC = fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
        if (isG && gfx < 0) gfx = i;
        if (isT && !isG && !isC && xferOnly < 0) xferOnly = i;
    }
    std::vector<uint32_t> use{ (uint32_t) gfx };
    if (xferOnly >= 0) use.push_back((uint32_t) xferOnly);
    float prio[1] = { 1.0f };
    std::vector<VkDeviceQueueCreateInfo> qis;
    for (uint32_t f : use) {
        VkDeviceQueueCreateInfo qi{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qi.queueFamilyIndex = f;
        qi.queueCount = 1;
        qi.pQueuePriorities = prio;
        qis.push_back(qi);
    }
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = (uint32_t) qis.size();
    dci.pQueueCreateInfos = qis.data();
    VK_CHECK(vkCreateDevice(g_phys, &dci, nullptr, &g_dev));
    vkGetDeviceQueue(g_dev, use[0], 0, &g_q0);
    g_q1 = g_q0;
    if (use.size() > 1) vkGetDeviceQueue(g_dev, use[1], 0, &g_q1);
    auto mkPool = [&](uint32_t fam) {
        VkCommandPoolCreateInfo cpi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = fam;
        VkCommandPool p;
        VK_CHECK(vkCreateCommandPool(g_dev, &cpi, nullptr, &p));
        return p;
    };
    g_pool0 = mkPool(use[0]);
    g_pool1 = use.size() > 1 ? mkPool(use[1]) : g_pool0;

    const VkMemoryPropertyFlags WC = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buf wcKeep, cachedKeep, devA, devB;
    createBuf(wcKeep, region, WC, "wcKeep");
    createBuf(cachedKeep, region, WC | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "cachedKeep");
    createBuf(devA, region, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "devA");
    createBuf(devB, region, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "devB");
    fillHost(wcKeep, true);
    fillHost(cachedKeep, true);

    makePipeline(argc > 2 ? argv[2] : "copy.spv");
    Desc dWcDev = makeDesc(wcKeep, devA);
    Desc dCaDev = makeDesc(cachedKeep, devA);
    Desc dDevWc = makeDesc(devB, wcKeep);
    Desc dDevDev = makeDesc(devB, devA);

    auto allocCb = [&](VkCommandPool pool) {
        VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = pool;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(g_dev, &cai, &cb));
        return cb;
    };
    auto kernelBW = [&](Desc & d, VkDeviceSize bytes, int wg, VkQueue q, VkCommandPool pool) {
        VkCommandBuffer cb = allocCb(pool);
        recordKernel(cb, d, bytes, wg);
        double sec = timedSubmit(cb, q);
        vkFreeCommandBuffers(g_dev, pool, 1, &cb);
        return (double) bytes / sec;
    };

    printf("\n=== SDMA reference (vkCmdCopyBuffer, single bulk submit) ===\n");
    printf("  H2D wc->dev       %8.2f GB/s\n", sdma(wcKeep, devA, region, g_q0, g_pool0) / GB);
    printf("  H2D cached->dev   %8.2f GB/s\n", sdma(cachedKeep, devA, region, g_q0, g_pool0) / GB);
    printf("  D2H dev->wc       %8.2f GB/s\n", sdma(devB, wcKeep, region, g_q0, g_pool0) / GB);
    printf("  D2D dev->dev      %8.2f GB/s\n", sdma(devB, devA, region, g_q0, g_pool0) / GB);

    printf("\n=== GPU-driven copy kernel (compute shader reads/writes across PCIe) ===\n");
    printf("%-22s %-8s %10s\n", "leg", "workgroups", "GB/s");
    for (int wg : { 16, 64, 256, 1024, 4096 }) {
        printf("  H2D wc->dev          %-8d %10.2f\n", wg, kernelBW(dWcDev, region, wg, g_q0, g_pool0) / GB);
    }
    for (int wg : { 256, 1024, 4096 }) {
        printf("  H2D cached->dev      %-8d %10.2f\n", wg, kernelBW(dCaDev, region, wg, g_q0, g_pool0) / GB);
        printf("  D2H dev->wc          %-8d %10.2f\n", wg, kernelBW(dDevWc, region, wg, g_q0, g_pool0) / GB);
        printf("  D2D dev->dev (kern)  %-8d %10.2f\n", wg, kernelBW(dDevDev, region, wg, g_q0, g_pool0) / GB);
    }

    printf("\n=== kernel + SDMA concurrently (same direction, then full duplex) ===\n");
    {
        auto leg = [&](std::function<double()> f, double * out) { *out = f(); };
        double a = 0, b = 0;
        auto t0 = std::chrono::steady_clock::now();
        std::thread t1(leg, [&] { return kernelBW(dWcDev, region, 1024, g_q0, g_pool0); }, &a);
        std::thread t2(leg, [&] { return sdma(cachedKeep, devB, region, g_q1, g_pool1); }, &b);
        t1.join();
        t2.join();
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("  2x H2D (kernel+sdma)  agg %6.2f GB/s   (%.2f + %.2f)\n", 2 * region / sec / GB, a / GB, b / GB);

        a = b = 0;
        t0 = std::chrono::steady_clock::now();
        std::thread t3(leg, [&] { return kernelBW(dWcDev, region, 1024, g_q0, g_pool0); }, &a);
        std::thread t4(leg, [&] { return sdma(devB, wcKeep, region, g_q1, g_pool1); }, &b);
        t3.join();
        t4.join();
        sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("  full duplex (kernel H2D + sdma D2H)  agg %6.2f GB/s   (%.2f out + %.2f in)\n",
               2 * region / sec / GB, a / GB, b / GB);
    }

    vkDestroyDevice(g_dev, nullptr);
    vkDestroyInstance(g_inst, nullptr);
    return 0;
}
