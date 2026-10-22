// Standalone Vulkan host<->device copy bandwidth probe.
// Establishes the real PCIe ceiling on this box and isolates the variables in
// the H2D path: host memory type, persistent mapping, queue family, chunk size,
// in-flight depth, direction mixing, and the pageable->pinned memcpy hop that
// ggml's staging path pays before the DMA.
// Build: see BUILD.txt in this directory.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define VK_CHECK(expr)                                                        \
    do {                                                                       \
        VkResult r_ = (expr);                                                  \
        if (r_ != VK_SUCCESS && r_ != VK_INCOMPLETE) {                          \
            printf("FAIL %s -> %d (line %d)\n", #expr, (int) r_, __LINE__);     \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static const double GB = 1e9;
static const double GiB = 1073741824.0;

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint8_t * persist = nullptr;
};

static VkInstance g_inst;
static VkPhysicalDevice g_phys;
static VkDevice g_dev;
static VkCommandPool g_pool;   // gfx family
static VkCommandPool g_pool2;  // transfer family (or g_pool)
static VkQueue g_q0, g_q1;
static std::vector<VkQueue> g_qv;
static std::vector<VkCommandPool> g_pv;
static std::vector<const char *> g_famName;

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
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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

static void destroyBuf(Buf & b) {
    if (b.persist) vkUnmapMemory(g_dev, b.mem);
    if (b.mem) vkFreeMemory(g_dev, b.mem, nullptr);
    if (b.buf) vkDestroyBuffer(g_dev, b.buf, nullptr);
    b = Buf{};
}

// Fill a host-visible buffer. If keepMapped, the mapping stays alive (the
// persistent-mapping case); otherwise it is unmapped so the copy reads a
// committed allocation (the map-then-free case).
static void fillHost(Buf & b, bool keepMapped) {
    void * p = nullptr;
    VK_CHECK(vkMapMemory(g_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &p));
    std::memset(p, 0x5a, (size_t) b.size);
    if (keepMapped) {
        b.persist = (uint8_t *) p;
    } else {
        vkUnmapMemory(g_dev, b.mem);
    }
}

static void mapThenFree(const Buf & b) {
    void * p = nullptr;
    VK_CHECK(vkMapMemory(g_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &p));
    vkUnmapMemory(g_dev, b.mem);
}

static void record(VkCommandBuffer cb, const Buf & src, const Buf & dst, VkDeviceSize off,
                   VkDeviceSize len, VkDeviceSize chunk, const Buf * src2, const Buf * dst2) {
    VkDeviceSize done = 0;
    while (done < len) {
        VkDeviceSize n = std::min(chunk, len - done);
        VkBufferCopy c{ off + done, off + done, n };
        vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &c);
        if (src2) vkCmdCopyBuffer(cb, src2->buf, dst2->buf, 1, &c);
        done += n;
    }
}

// One timed sequence over `total` bytes.
// mode: "bulk" (1 submit) | "serial" (submit+wait per chunk) | "pipe" (K in flight)
static double transfer(const char * dir, const Buf & src, const Buf & dst, VkDeviceSize total,
                       VkDeviceSize chunk, const char * mode, int K, VkQueue q, VkCommandPool pool,
                       bool touchMap, const Buf * revSrc = nullptr, const Buf * revDst = nullptr) {
    if (touchMap) mapThenFree(src);

    int segs = (std::strcmp(mode, "bulk") == 0) ? 1 : (int) ((total + chunk - 1) / chunk);
    if (std::strcmp(mode, "serial") == 0) K = 1;
    if (std::strcmp(mode, "bulk") == 0) K = 1;

    std::vector<VkCommandBuffer> cbs(segs);
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = segs;
    VK_CHECK(vkAllocateCommandBuffers(g_dev, &cai, cbs.data()));
    std::vector<VkFence> fences(segs);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    for (int i = 0; i < segs; i++) {
        VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(g_dev, &fi, nullptr, &fences[i]));
        VK_CHECK(vkBeginCommandBuffer(cbs[i], &bi));
        VkDeviceSize off = (std::strcmp(mode, "bulk") == 0) ? 0 : (VkDeviceSize) i * chunk;
        VkDeviceSize len = (std::strcmp(mode, "bulk") == 0) ? total : std::min(chunk, total - off);
        record(cbs[i], src, dst, off, len, chunk, revSrc, revDst);
        VK_CHECK(vkEndCommandBuffer(cbs[i]));
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < segs; s += K) {
        int n = std::min(K, segs - s);
        for (int i = 0; i < n; i++) {
            VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cbs[s + i];
            VK_CHECK(vkQueueSubmit(q, 1, &si, fences[s + i]));
        }
        VK_CHECK(vkWaitForFences(g_dev, n, &fences[s], VK_TRUE, ~0ull));
    }
    auto t1 = std::chrono::steady_clock::now();

    for (auto f : fences) vkDestroyFence(g_dev, f, nullptr);
    vkFreeCommandBuffers(g_dev, pool, segs, cbs.data());
    double sec = std::chrono::duration<double>(t1 - t0).count();
    return (double) total * (revSrc ? 2.0 : 1.0) / sec;
}

// Realistic ggml staging path: pageable RAM -> pinned ring -> device.
// memcpy of window i+1 overlaps the DMA of window i, ring depth = K.
static double viaStaging(const Buf & stage, const Buf & dst, const uint8_t * pageable,
                         VkDeviceSize total, VkDeviceSize chunk, int K, VkQueue q, VkCommandPool pool,
                         double * memcpyOnly) {
    const VkDeviceSize ring = chunk * K;
    const int segs = (int) ((total + chunk - 1) / chunk);
    std::vector<VkCommandBuffer> cbs(segs);
    std::vector<VkFence> fences(segs);
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = pool;
    cai.commandBufferCount = segs;
    VK_CHECK(vkAllocateCommandBuffers(g_dev, &cai, cbs.data()));
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    for (int i = 0; i < segs; i++) {
        VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(g_dev, &fi, nullptr, &fences[i]));
        VK_CHECK(vkBeginCommandBuffer(cbs[i], &bi));
        VkDeviceSize off = (VkDeviceSize) i * chunk;
        VkDeviceSize len = std::min(chunk, total - off);
        VkBufferCopy c{ off % ring, off, len };
        vkCmdCopyBuffer(cbs[i], stage.buf, dst.buf, 1, &c);
        VK_CHECK(vkEndCommandBuffer(cbs[i]));
    }

    double memcpySec = 0;
    auto fill = [&](int i) {
        VkDeviceSize off = (VkDeviceSize) i * chunk;
        auto m0 = std::chrono::steady_clock::now();
        std::memcpy(stage.persist + (off % ring), pageable + off, std::min(chunk, total - off));
        memcpySec += std::chrono::duration<double>(std::chrono::steady_clock::now() - m0).count();
    };
    auto submit = [&](int i) {
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cbs[i];
        VK_CHECK(vkQueueSubmit(q, 1, &si, fences[i]));
    };

    auto t0 = std::chrono::steady_clock::now();
    int prologue = std::min(K, segs);
    for (int i = 0; i < prologue; i++) { fill(i); submit(i); }
    for (int i = K; i < segs; i++) {
        VK_CHECK(vkWaitForFences(g_dev, 1, &fences[i - K], VK_TRUE, ~0ull));
        fill(i);
        submit(i);
    }
    int tail = std::min(K, segs);
    VK_CHECK(vkWaitForFences(g_dev, tail, &fences[segs - tail], VK_TRUE, ~0ull));
    auto t1 = std::chrono::steady_clock::now();

    for (auto f : fences) vkDestroyFence(g_dev, f, nullptr);
    vkFreeCommandBuffers(g_dev, pool, segs, cbs.data());
    if (memcpyOnly) *memcpyOnly = (double) total / memcpySec / GB;
    return (double) total / std::chrono::duration<double>(t1 - t0).count();
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
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    double devLocalGiB = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            devLocalGiB += mp.memoryHeaps[i].size / GiB;
    printf("device: %s  api %u.%u  nonCoherentAtomSize=%llu  deviceLocalHeap=%.2f GiB\n",
           pprops.deviceName, VK_VERSION_MAJOR(pprops.apiVersion), VK_VERSION_MINOR(pprops.apiVersion),
           (unsigned long long) pprops.limits.nonCoherentAtomSize, devLocalGiB);

    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &n, fams.data());
    int gfx = -1, xfer = -1;
    for (uint32_t i = 0; i < n; i++) {
        bool isG = fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
        bool isT = fams[i].queueFlags & VK_QUEUE_TRANSFER_BIT;
        printf("  queue fam %u: flags=0x%x count=%u\n", i, fams[i].queueFlags, fams[i].queueCount);
        if (isG && gfx < 0) gfx = i;
        if (isT && !isG && xfer < 0) xfer = i;
    }

    std::vector<uint32_t> use{ (uint32_t) gfx };
    for (uint32_t i = 0; i < n; i++) {
        if ((fams[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && (int) i != gfx) use.push_back(i);
    }
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
    g_pool = mkPool(use[0]);
    g_pool2 = use.size() > 1 ? mkPool(use[1]) : g_pool;
    for (uint32_t f : use) {
        VkQueue q;
        vkGetDeviceQueue(g_dev, f, 0, &q);
        g_qv.push_back(q);
        g_pv.push_back(mkPool(f));
        g_famName.push_back(f == (uint32_t) gfx
                                ? "gfx"
                                : ((fams[f].queueFlags & VK_QUEUE_COMPUTE_BIT) ? "cmp+xfr" : "xfer-only"));
    }
    {
        uint32_t t = findMemoryType(0xFFFFFFFFu, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        printf("  resizable-BAR style host-visible+device-local memory type: %s\n",
               t == 0xFFFFFFFFu ? "none" : "present");
    }

    const VkMemoryPropertyFlags WC = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buf wcFree, wcKeep, cachedFree, cachedKeep, devA, devB, devC, staging;
    createBuf(wcFree, region, WC, "wcFree");
    createBuf(wcKeep, region, WC, "wcKeep");
    createBuf(cachedFree, region, WC | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "cachedFree");
    createBuf(cachedKeep, region, WC | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "cachedKeep");
    createBuf(devA, region, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "devA");
    createBuf(devB, region, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "devB");
    createBuf(devC, region, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "devC");
    createBuf(staging, 256ull << 20, WC | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "staging");
    fillHost(wcFree, false);
    fillHost(cachedFree, false);
    fillHost(wcKeep, true);
    fillHost(cachedKeep, true);
    void * sp = nullptr;
    VK_CHECK(vkMapMemory(g_dev, staging.mem, 0, VK_WHOLE_SIZE, 0, &sp));
    staging.persist = (uint8_t *) sp;
    std::memset(staging.persist, 0, (size_t) staging.size);
    // prime devB so D2H has real device-side data
    transfer("H2D", wcFree, devB, region, region, "bulk", 1, g_q0, g_pool, false);

    struct Row { const char * tag; Buf * src; Buf * dst; bool touch; const char * ql; VkQueue q; VkCommandPool p; };
    std::vector<Row> rows = {
        { "H2D wc/free", &wcFree, &devA, true, "gfx", g_q0, g_pool },
        { "H2D wc/keep", &wcKeep, &devA, false, "gfx", g_q0, g_pool },
        { "H2D cached/free", &cachedFree, &devA, true, "gfx", g_q0, g_pool },
        { "H2D cached/keep", &cachedKeep, &devA, false, "gfx", g_q0, g_pool },
    };
    if (xfer >= 0) {
        rows.push_back({ "H2D wc/keep", &wcKeep, &devA, false, "xfer", g_q1, g_pool2 });
        rows.push_back({ "H2D cached/keep", &cachedKeep, &devA, false, "xfer", g_q1, g_pool2 });
    }

    printf("\n=== ceiling sweep (%.0f MiB per test) ===\n", region / 1048576.0);
    printf("%-20s %-5s %-5s %-6s %-4s %9s\n", "path", "queue", "chunk", "mode", "K", "GB/s");
    printf("-------------------------------------------------------------------------\n");
    struct T { VkDeviceSize chunk; const char * cname; const char * mode; int K; };
    std::vector<T> ts = {
        { region, "full", "bulk", 1 },
        { 64ull << 20, "64M", "pipe", 8 },
        { 16ull << 20, "16M", "pipe", 8 },
        { 16ull << 20, "16M", "pipe", 4 },
        { 16ull << 20, "16M", "pipe", 1 },
        { 16ull << 20, "16M", "serial", 1 },
        { 4ull << 20, "4M", "pipe", 8 },
        { 4ull << 20, "4M", "serial", 1 },
        { 1536ull << 10, "1.5M", "pipe", 8 },
        { 1536ull << 10, "1.5M", "serial", 1 },
    };
    for (auto & r : rows)
        for (auto & t : ts)
            printf("%-20s %-5s %-5s %-6s %-4d %9.2f\n", r.tag, r.ql, t.cname, t.mode, t.K,
                   transfer(r.tag, *r.src, *r.dst, region, t.chunk, t.mode, t.K, r.q, r.p, r.touch) / GB);

    printf("\n=== in-flight depth, H2D wc/keep gfx ===\n");
    for (VkDeviceSize ch : { 4ull << 20, 16ull << 20 })
        for (int K : { 1, 2, 4, 8, 16, 32, 64 }) {
            double bw = transfer("H2D", wcKeep, devA, region, ch, "pipe", K, g_q0, g_pool, false);
            printf("  chunk %-4llu MiB K=%-3d %8.2f GB/s  %8.2f GiB/s\n",
                   (unsigned long long) (ch >> 20), K, bw / GB, bw / GiB);
        }

    printf("\n=== other legs ===\n");
    printf("  D2H  16M pipe K=8   %8.2f GB/s\n",
           transfer("D2H", devB, wcFree, region, 16 << 20, "pipe", 8, g_q0, g_pool, false) / GB);
    printf("  D2H  full bulk      %8.2f GB/s\n",
           transfer("D2H", devB, wcFree, region, region, "bulk", 1, g_q0, g_pool, false) / GB);
    printf("  H2D+D2H bidir K=8   %8.2f GB/s (sum of both)\n",
           transfer("H2D", wcKeep, devA, region, 16 << 20, "pipe", 8, g_q0, g_pool, false, &devB, &cachedKeep) / GB);
    {
        double bw = transfer("D2D", devB, devA, region, region, "bulk", 1, g_q0, g_pool, false);
        printf("  D2D  full bulk      %8.2f GB/s copied (%.2f GB/s VRAM r+w)\n", bw / GB, 2 * bw / GB);
    }

    printf("\n=== concurrent legs: is the ceiling the wire, or one copy engine? ===\n");
    {
        struct Leg { Buf * src; Buf * dst; int fam; };
        auto runLegs = [&](std::vector<Leg> legs, const char * label) {
            std::vector<double> per(legs.size(), 0);
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (size_t i = 0; i < legs.size(); i++)
                th.emplace_back([&, i] {
                    per[i] = transfer("x", *legs[i].src, *legs[i].dst, region, 16ull << 20, "pipe", 8,
                                      g_qv[legs[i].fam], g_pv[legs[i].fam], false);
                });
            for (auto & x : th) x.join();
            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            printf("  %-38s wall %.3f s  aggregate %6.2f GB/s   legs:", label, sec,
                   (double) region * legs.size() / sec / GB);
            for (size_t i = 0; i < legs.size(); i++) printf(" %.2f", per[i] / GB);
            printf("\n");
            fflush(stdout);
        };
        runLegs({ { &wcKeep, &devA, 0 } }, "1 leg H2D (gfx)");
        runLegs({ { &wcKeep, &devA, 0 }, { &cachedKeep, &devB, 1 } }, "2 legs H2D (gfx + fam1)");
        if (g_qv.size() > 2)
            runLegs({ { &wcKeep, &devA, 0 }, { &cachedKeep, &devB, 1 }, { &wcFree, &devC, 2 } },
                    "3 legs H2D (gfx + fam1 + fam2)");
        runLegs({ { &wcKeep, &devA, 0 }, { &devB, &cachedFree, 1 } }, "H2D (gfx) + D2H (fam1) full duplex");
        if (g_qv.size() > 2)
            runLegs({ { &wcKeep, &devA, 0 }, { &devB, &cachedFree, 2 } },
                    "H2D (gfx) + D2H (xfer-only) full duplex");
        if (g_qv.size() > 2)
            runLegs({ { &wcKeep, &devA, 1 }, { &cachedKeep, &devB, 2 } }, "2 legs H2D (fam1 + fam2, no gfx)");
    }

    printf("\n=== H2D under host-RAM saturation (does the DMA path need a CPU copy?) ===\n");
    {
        std::atomic<bool> stop{ false };
        const size_t per = 8u << 20;
        std::vector<double> ram(8, 0);
        std::vector<std::thread> th;
        double idle = transfer("x", wcKeep, devA, region, 16ull << 20, "pipe", 8, g_q0, g_pool, false);
        for (int i = 0; i < 8; i++)
            th.emplace_back([&, i] {
                std::vector<uint8_t> a(per, 1), b(per);
                double acc = 0;
                size_t done = 0;
                while (!stop.load()) {
                    auto s0 = std::chrono::steady_clock::now();
                    std::memcpy(b.data(), a.data(), per);
                    auto s1 = std::chrono::steady_clock::now();
                    done += per;
                    acc += std::chrono::duration<double>(s1 - s0).count();
                }
                ram[i] = done / acc / GB;
            });
        double busy = transfer("x", wcKeep, devA, region, 16ull << 20, "pipe", 8, g_q0, g_pool, false);
        stop = true;
        for (auto & x : th) x.join();
        double sum = 0;
        for (double r : ram) sum += r;
        printf("  H2D idle %.2f GB/s -> under 8-thread memcpy load %.2f GB/s (memcpy aggregate %.2f GB/s)\n",
               idle / GB, busy / GB, sum);
    }

    printf("\n=== pageable RAM -> pinned staging -> device (ggml staging shape) ===\n");
    std::vector<uint8_t> pageable((size_t) region, 0x33);
    for (VkDeviceSize ch : { 4ull << 20, 16ull << 20 })
        for (int K : { 1, 2, 4, 8 }) {
            double mo = 0;
            if ((VkDeviceSize) (staging.size / ch) < K) continue;
            double bw = viaStaging(staging, devA, pageable.data(), region, ch, K, g_q0, g_pool, &mo);
            printf("  chunk %-3llu MiB K=%d  end-to-end %8.2f GB/s   (memcpy alone %6.2f GB/s)\n",
                   (unsigned long long) (ch >> 20), K, bw / GB, mo);
        }

    {
        size_t bytes = (size_t) region;
        std::vector<uint8_t> b(bytes);
        double best = 0;
        for (int i = 0; i < 3; i++) {
            auto t0 = std::chrono::steady_clock::now();
            std::memcpy(b.data(), pageable.data(), bytes);
            auto t1 = std::chrono::steady_clock::now();
            best = std::max(best, bytes / std::chrono::duration<double>(t1 - t0).count() / GB);
        }
        printf("  host memcpy 1 thread (pageable->pageable) %8.2f GB/s of traffic\n", best);
        double mt = 0;
        {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            int T = 8;
            for (int i = 0; i < T; i++)
                th.emplace_back([&, i] {
                    size_t per = bytes / T;
                    std::memcpy(b.data() + i * per, pageable.data() + i * per, per);
                });
            for (auto & x : th) x.join();
            auto t1 = std::chrono::steady_clock::now();
            mt = bytes / std::chrono::duration<double>(t1 - t0).count() / GB;
        }
        printf("  host memcpy 8 threads                     %8.2f GB/s of traffic\n", mt);
    }

    destroyBuf(wcFree); destroyBuf(wcKeep); destroyBuf(cachedFree); destroyBuf(cachedKeep);
    destroyBuf(devA); destroyBuf(devB); destroyBuf(devC); destroyBuf(staging);
    vkDestroyDevice(g_dev, nullptr);
    vkDestroyInstance(g_inst, nullptr);
    return 0;
}
