#include "vk_dev.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef QRT_HAVE_VULKAN

namespace {

const uint64_t MIB = 1024ull * 1024ull;

// Every device buffer's size, so a copy that runs past its buffer is a reported error
// instead of a driver fault (RESULTS 75.5 defect 2: a 524 KiB download from a 144 KiB
// plane ended in VK_ERROR_DEVICE_LOST). Handles are unique while the device lives.
enum { QRT_BUF_REG = 4096 };
struct qrt_buf_entry {
    VkBuffer buf;
    uint64_t bytes;
};
qrt_buf_entry g_buf_reg[QRT_BUF_REG];

void buf_reg_add(VkBuffer buf, uint64_t bytes) {
    for (uint32_t i = 0; i < QRT_BUF_REG; ++i) {
        if (g_buf_reg[i].buf == VK_NULL_HANDLE || g_buf_reg[i].buf == buf) {
            g_buf_reg[i].buf = buf;
            g_buf_reg[i].bytes = bytes;
            return;
        }
    }
}

void buf_reg_del(VkBuffer buf) {
    for (uint32_t i = 0; i < QRT_BUF_REG; ++i) {
        if (g_buf_reg[i].buf == buf) {
            g_buf_reg[i].buf = VK_NULL_HANDLE;
            g_buf_reg[i].bytes = 0;
            return;
        }
    }
}

// An unregistered handle is not rejected: modules may be handed buffers from a pool this
// registry never saw. The check only fires where the size is known.
int fail(char * err, size_t err_size, const char * fmt, ...);

int buf_reg_check(VkBuffer buf, uint64_t offset, uint64_t bytes, char * err, size_t err_size) {
    for (uint32_t i = 0; i < QRT_BUF_REG; ++i) {
        if (g_buf_reg[i].buf != buf) {
            continue;
        }
        if (offset > g_buf_reg[i].bytes || bytes > g_buf_reg[i].bytes - offset) {
            return fail(err, err_size, "copy of %llu bytes at %llu runs past a %llu byte buffer",
                        (unsigned long long) bytes, (unsigned long long) offset,
                        (unsigned long long) g_buf_reg[i].bytes);
        }
        return 0;
    }
    return 0;
}

// design 5.5 buckets, with the measured bytes from RESULTS section 25:
// dense (non-expert, incl. output head) 2180153856 B = 2079 MiB, draft 100 MiB,
// compute 700 MiB is the design's lower bound (decode tensors are only 9.1 MiB).
const uint64_t DENSE_BYTES   = 2079 * MIB;
const uint64_t DRAFT_BYTES   = 100 * MIB;
const uint64_t COMPUTE_BYTES = 700 * MIB;
const uint64_t CACHE_STEP    = 256 * MIB;
const uint64_t DESKTOP_BYTES = 1000 * MIB;

int fail(char * err, size_t err_size, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_size, fmt, ap);
    va_end(ap);
    return -1;
}

uint32_t pick_heap(const VkPhysicalDeviceMemoryProperties * mem, uint64_t * size) {
    uint32_t best = UINT32_MAX;
    *size = 0;
    for (uint32_t i = 0; i < mem->memoryHeapCount; ++i) {
        if ((mem->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            mem->memoryHeaps[i].size > *size) {
            best = i;
            *size = mem->memoryHeaps[i].size;
        }
    }
    return best;
}

uint32_t pick_memory_type(const qrt_vk_dev * dev, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < dev->mem.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (dev->mem.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    for (uint32_t i = 0; i < dev->mem.memoryTypeCount; ++i) {
        if (bits & (1u << i)) {
            return i;
        }
    }
    return UINT32_MAX;
}

// Unlike pick_memory_type this never falls back to a type that cannot be mapped:
// mapping non-host-visible memory is a validation error, and on Windows it faults.
// Cached system memory wins because the CPU reads the mapping (RESULTS 35): reading
// a write-combined mapping runs at 0.21 GB/s, the cached one at PCIe speed, and the
// H2D rate is the same for both. *coherent is 0 when only a non-coherent type fits,
// so the caller flushes and invalidates explicitly.
uint32_t pick_host_type(const qrt_vk_dev * dev, uint32_t bits, int * coherent) {
    static const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags tries[] = {
        host | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        host | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        host | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        host,
    };
    for (int t = 0; t < 4; ++t) {
        for (uint32_t i = 0; i < dev->mem.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (dev->mem.memoryTypes[i].propertyFlags & tries[t]) == tries[t]) {
                *coherent = (tries[t] & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 1 : 0;
                return i;
            }
        }
    }
    *coherent = 0;
    return UINT32_MAX;
}

int query_budget(qrt_vk_dev * dev, char * err, size_t err_size) {
    uint64_t heap_size = 0;
    const uint32_t heap = pick_heap(&dev->mem, &heap_size);
    if (heap == UINT32_MAX) {
        return fail(err, err_size, "no device-local memory heap");
    }
    dev->heap_bytes = heap_size;
    uint64_t budget = heap_size;
    dev->usage_bytes = 0;
    if (dev->has_memory_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp;
        memset(&bp, 0, sizeof(bp));
        bp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp;
        memset(&mp, 0, sizeof(mp));
        mp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mp.pNext = &bp;
        vkGetPhysicalDeviceMemoryProperties2(dev->phys, &mp);
        // heapBudget already accounts for other processes, so do not subtract the desktop again.
        budget = bp.heapBudget[heap];
        dev->usage_bytes = bp.heapUsage[heap];
    } else if (heap_size > DESKTOP_BYTES) {
        budget = heap_size - DESKTOP_BYTES;
    } else {
        budget = 0;
    }
    if (dev->tier_mib != 0) {
        const uint64_t tier = (uint64_t) dev->tier_mib * MIB;
        if (tier < budget) {
            budget = tier;
        }
    }
    dev->budget_bytes = budget;
    return 0;
}

bool has_device_extension(VkPhysicalDevice phys, const char * name) {
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr) != VK_SUCCESS || n == 0) {
        return false;
    }
    VkExtensionProperties * ext = (VkExtensionProperties *) calloc(n, sizeof(*ext));
    if (ext == NULL) {
        return false;
    }
    const bool ok = vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, ext) == VK_SUCCESS;
    bool found = false;
    for (uint32_t i = 0; ok && i < n; ++i) {
        if (strcmp(ext[i].extensionName, name) == 0) {
            found = true;
            break;
        }
    }
    free(ext);
    return found;
}

} // namespace

int qrt_vk_open(uint32_t tier_mib, qrt_vk_dev ** out, char * err, size_t err_size) {
    *out = NULL;
    qrt_vk_dev * dev = new qrt_vk_dev();
    memset(dev, 0, sizeof(*dev));
    dev->tier_mib = tier_mib;

    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "qrt";
    app.apiVersion         = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkResult r = vkCreateInstance(&ici, nullptr, &dev->instance);
    if (r != VK_SUCCESS) {
        fail(err, err_size, "vkCreateInstance failed: VkResult %d", (int) r);
        delete dev;
        return -1;
    }

    uint32_t n_dev = 0;
    r = vkEnumeratePhysicalDevices(dev->instance, &n_dev, nullptr);
    if (r != VK_SUCCESS || n_dev == 0) {
        fail(err, err_size, "no Vulkan device (VkResult %d, %u devices)", (int) r, n_dev);
        qrt_vk_close(dev);
        return -1;
    }
    VkPhysicalDevice * phys = (VkPhysicalDevice *) calloc(n_dev, sizeof(*phys));
    if (phys == NULL) {
        fail(err, err_size, "out of memory for %u devices", n_dev);
        qrt_vk_close(dev);
        return -1;
    }
    vkEnumeratePhysicalDevices(dev->instance, &n_dev, phys);
    // Prefer a discrete GPU, then the largest device-local heap.
    uint64_t best_score = 0;
    for (uint32_t i = 0; i < n_dev; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(phys[i], &p);
        VkPhysicalDeviceMemoryProperties m;
        vkGetPhysicalDeviceMemoryProperties(phys[i], &m);
        uint64_t heap = 0;
        pick_heap(&m, &heap);
        const uint64_t score = heap + (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? (1ull << 40) : 0);
        if (dev->phys == VK_NULL_HANDLE || score > best_score) {
            dev->phys = phys[i];
            best_score = score;
        }
    }
    free(phys);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev->phys, &props);
    snprintf(dev->name, sizeof(dev->name), "%s", props.deviceName);

    uint32_t n_fam = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev->phys, &n_fam, nullptr);
    VkQueueFamilyProperties * fam = (VkQueueFamilyProperties *) calloc(n_fam, sizeof(*fam));
    if (fam == NULL) {
        fail(err, err_size, "out of memory for %u queue families", n_fam);
        qrt_vk_close(dev);
        return -1;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(dev->phys, &n_fam, fam);
    dev->compute_family  = UINT32_MAX;
    dev->transfer_family = UINT32_MAX;
    for (uint32_t i = 0; i < n_fam; ++i) {
        const VkQueueFlags f = fam[i].queueFlags;
        if ((f & VK_QUEUE_COMPUTE_BIT) && (f & VK_QUEUE_GRAPHICS_BIT) == 0 &&
            dev->compute_family == UINT32_MAX) {
            dev->compute_family = i;
        }
    }
    for (uint32_t i = 0; i < n_fam && dev->compute_family == UINT32_MAX; ++i) {
        if (fam[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            dev->compute_family = i;
        }
    }
    for (uint32_t i = 0; i < n_fam; ++i) {
        const VkQueueFlags f = fam[i].queueFlags;
        if ((f & VK_QUEUE_TRANSFER_BIT) && (f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0) {
            dev->transfer_family = i;
            break;
        }
    }
    if (dev->transfer_family == UINT32_MAX) {
        dev->transfer_family = dev->compute_family;
    }
    if (dev->compute_family == UINT32_MAX) {
        fail(err, err_size, "'%s' has no compute queue", dev->name);
        free(fam);
        qrt_vk_close(dev);
        return -1;
    }
    dev->timestamp_bits = fam[dev->compute_family].timestampValidBits;
    free(fam);

    dev->has_memory_budget = has_device_extension(dev->phys, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) ? 1 : 0;

    // int8 dot product (VK_KHR_shader_integer_dot_product, core in 1.3): the Q4_K x Q8_1
    // GEMM is 3.66 TFLOPS without it and 6.08-6.20 with it (RESULTS 28). Optional: a
    // device without it still opens, and the kernel reports the missing feature.
    dev->has_dot_product = 0;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dotf;
    memset(&dotf, 0, sizeof(dotf));
    dotf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    VkPhysicalDeviceVulkan12Features f12;
    memset(&f12, 0, sizeof(f12));
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.pNext = &dotf;
    VkPhysicalDeviceFeatures2 f2;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(dev->phys, &f2);
    if (!f12.timelineSemaphore) {
        fail(err, err_size, "'%s' has no timeline semaphore (Vulkan 1.2)", dev->name);
        qrt_vk_close(dev);
        return -1;
    }
    const int dot_ext = has_device_extension(dev->phys, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) ? 1 : 0;
    dev->has_dot_product = (dotf.shaderIntegerDotProduct == VK_TRUE) ? 1 : 0;

    // Subgroup size is not 32 everywhere (RDNA can report wave64 and some iGPUs 8
    // or 16), and the DeltaNet recurrence reduces across it, so kernels specialize
    // on the real value instead of assuming it.
    {
        VkPhysicalDeviceSubgroupProperties sub;
        memset(&sub, 0, sizeof(sub));
        sub.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 p2;
        memset(&p2, 0, sizeof(p2));
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &sub;
        vkGetPhysicalDeviceProperties2(dev->phys, &p2);
        dev->subgroup_size = sub.subgroupSize;
        dev->has_subgroup_arithmetic =
            (sub.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? 1 : 0;
    }
    VkPhysicalDeviceVulkan12Features enable12;
    memset(&enable12, 0, sizeof(enable12));
    enable12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enable12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceShaderIntegerDotProductFeatures enable_dot;
    memset(&enable_dot, 0, sizeof(enable_dot));
    enable_dot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    enable_dot.shaderIntegerDotProduct = VK_TRUE;
    if (dev->has_dot_product && dot_ext) {
        enable12.pNext = &enable_dot;
    } else if (dev->has_dot_product && !dot_ext) {
        // Promoted to core only from 1.3, so without the extension it is unusable here.
        dev->has_dot_product = 0;
    }

    VkDeviceQueueCreateInfo qci[2];
    memset(qci, 0, sizeof(qci));
    const float prio = 1.0f;
    uint32_t n_qci = 0;
    qci[n_qci].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci[n_qci].queueFamilyIndex = dev->compute_family;
    qci[n_qci].queueCount       = 1;
    qci[n_qci].pQueuePriorities = &prio;
    ++n_qci;
    if (dev->transfer_family != dev->compute_family) {
        qci[n_qci].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci[n_qci].queueFamilyIndex = dev->transfer_family;
        qci[n_qci].queueCount       = 1;
        qci[n_qci].pQueuePriorities = &prio;
        ++n_qci;
    }

    const char * ext_names[2];
    uint32_t n_ext = 0;
    if (dev->has_memory_budget) {
        ext_names[n_ext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
    }
    if (dev->has_dot_product) {
        ext_names[n_ext++] = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
    }
    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext                   = &enable12;
    dci.queueCreateInfoCount    = n_qci;
    dci.pQueueCreateInfos       = qci;
    dci.enabledExtensionCount   = n_ext;
    dci.ppEnabledExtensionNames = ext_names;
    r = vkCreateDevice(dev->phys, &dci, nullptr, &dev->device);
    if (r != VK_SUCCESS) {
        fail(err, err_size, "vkCreateDevice failed: VkResult %d", (int) r);
        qrt_vk_close(dev);
        return -1;
    }
    vkGetDeviceQueue(dev->device, dev->compute_family, 0, &dev->compute_queue);
    if (dev->transfer_family == dev->compute_family) {
        dev->transfer_queue = dev->compute_queue;
    } else {
        vkGetDeviceQueue(dev->device, dev->transfer_family, 0, &dev->transfer_queue);
    }
    vkGetPhysicalDeviceMemoryProperties(dev->phys, &dev->mem);

    if (query_budget(dev, err, err_size) != 0) {
        qrt_vk_close(dev);
        return -1;
    }
    *out = dev;
    return 0;
}

void qrt_vk_close(qrt_vk_dev * dev) {
    if (dev == NULL) {
        return;
    }
    if (dev->device != VK_NULL_HANDLE) {
        vkDestroyDevice(dev->device, nullptr);
    }
    if (dev->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(dev->instance, nullptr);
    }
    delete dev;
}

int qrt_vk_query_budget(qrt_vk_dev * dev, char * err, size_t err_size) {
    return query_budget(dev, err, err_size);
}

uint32_t qrt_vk_kv_window_tokens(const qrt_vk_dev * dev, const qrt_config * cfg,
                                 uint64_t kv_bytes_per_token, uint32_t n_ctx) {
    const uint64_t fixed = DENSE_BYTES + COMPUTE_BYTES + DRAFT_BYTES;
    if (dev->budget_bytes <= fixed || kv_bytes_per_token == 0 || n_ctx == 0) {
        return 0;
    }
    const uint64_t avail = dev->budget_bytes - fixed;
    const uint64_t page = cfg->kv_page_tokens > 0 ? (uint64_t) cfg->kv_page_tokens : 16384ull;
    const uint64_t page_bytes = page * kv_bytes_per_token;
    const uint64_t cache_want = (uint64_t) cfg->expert_cache_mib * MIB;
    uint64_t cache = avail > cache_want ? cache_want : avail;
    // the KV window keeps at least one page, and takes it out of the cache if it must
    if (avail - cache < page_bytes) {
        cache = avail > page_bytes ? avail - page_bytes : 0;
    }
    const uint64_t kv_avail = avail - cache;
    uint64_t pages = kv_avail / page_bytes;
    const uint64_t want_pages = ((uint64_t) n_ctx + page - 1ull) / page;
    if (pages > want_pages) {
        pages = want_pages;
    }
    uint64_t tokens = pages * page;
    if (tokens > n_ctx) {
        tokens = n_ctx;
    }
    if (tokens == 0 && (uint64_t) n_ctx * kv_bytes_per_token <= kv_avail) {
        tokens = n_ctx;
    }
    return (uint32_t) tokens;
}

int qrt_vk_plan_budget(qrt_vk_dev * dev, const qrt_config * cfg, uint64_t expert_store_bytes,
                qrt_vk_plan * out, char * err, size_t err_size) {
    memset(out, 0, sizeof(*out));
    // 11 full-attention layers, n_kv = 2, head_dim = 256 (design 5.5): the same
    // accounting the attention module allocates per token.
    const uint32_t row_k = qrt_kv_row_bytes(cfg->kv_k, 256);
    const uint32_t row_v = qrt_kv_row_bytes(cfg->kv_v, 256);
    if (row_k == 0 || row_v == 0) {
        return fail(err, err_size, "KV precision K=%d V=%d has no row layout",
                    cfg->kv_k, cfg->kv_v);
    }
    const uint64_t kv_per_token = 11ull * 2ull * (uint64_t) (row_k + row_v);
    // The window is whatever is left after the other buckets and the cache target, capped at
    // the context: a 16 GiB card gets all of it, an 8 GiB card pages the rest from host.
    const uint32_t window = qrt_vk_kv_window_tokens(dev, cfg, kv_per_token, QRT_KV_CTX_TOKENS);
    const uint64_t kv_bytes = (uint64_t) window * kv_per_token;
    out->dense_bytes   = DENSE_BYTES;
    out->kv_bytes      = kv_bytes;
    out->kv_tokens     = window;
    out->compute_bytes = COMPUTE_BYTES;
    out->draft_bytes   = DRAFT_BYTES;

    const uint64_t fixed = DENSE_BYTES + kv_bytes + COMPUTE_BYTES + DRAFT_BYTES;
    if (fixed > dev->budget_bytes) {
        dev->cache_reductions = 0;
        out->total_bytes = fixed;
        return fail(err, err_size,
                    "the fixed buckets need %llu MiB but the budget is %llu MiB",
                    (unsigned long long) (fixed / MIB),
                    (unsigned long long) (dev->budget_bytes / MIB));
    }
    const uint64_t avail = dev->budget_bytes - fixed;
    const uint64_t want = (uint64_t) cfg->expert_cache_mib * MIB;
    // Shrink in 256 MiB steps for the counter, then grant every whole MiB that is left.
    uint64_t cache = want < avail ? want : avail;
    cache = cache / MIB * MIB;
    uint32_t reductions = 0;
    if (want > cache) {
        reductions = (uint32_t) ((want - cache + CACHE_STEP - 1) / CACHE_STEP);
    }

    out->expert_cache_bytes = cache;
    out->cache_reductions   = reductions;
    out->total_bytes        = fixed + cache;
    if (expert_store_bytes > 0) {
        out->cache_coverage_x100 = (uint32_t) (cache * 10000ull / expert_store_bytes);
    }
    dev->cache_reductions = reductions;
    return 0;
}

int qrt_vk_alloc(qrt_vk_dev * dev, uint64_t bytes, VkBufferUsageFlags usage,
                 VkBuffer * buf, VkDeviceMemory * mem, char * err, size_t err_size) {
    *buf = VK_NULL_HANDLE;
    *mem = VK_NULL_HANDLE;
    if (bytes == 0) {
        ++dev->alloc_failures;
        return fail(err, err_size, "allocation of 0 bytes");
    }
    if (bytes > dev->budget_bytes) {
        ++dev->alloc_failures;
        return fail(err, err_size, "allocation of %llu MiB is above the %llu MiB budget",
                    (unsigned long long) (bytes / MIB),
                    (unsigned long long) (dev->budget_bytes / MIB));
    }
    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = bytes;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(dev->device, &bci, nullptr, buf);
    if (r != VK_SUCCESS) {
        ++dev->alloc_failures;
        return fail(err, err_size, "vkCreateBuffer of %llu MiB failed: VkResult %d",
                    (unsigned long long) (bytes / MIB), (int) r);
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev->device, *buf, &req);
    const uint32_t type = pick_memory_type(dev, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) {
        vkDestroyBuffer(dev->device, *buf, nullptr);
        *buf = VK_NULL_HANDLE;
        ++dev->alloc_failures;
        return fail(err, err_size, "no memory type fits a %llu MiB buffer",
                    (unsigned long long) (bytes / MIB));
    }
    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = type;
    r = vkAllocateMemory(dev->device, &mai, nullptr, mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(dev->device, *buf, nullptr);
        *buf = VK_NULL_HANDLE;
        ++dev->alloc_failures;
        return fail(err, err_size, "vkAllocateMemory of %llu MiB failed: VkResult %d",
                    (unsigned long long) (req.size / MIB), (int) r);
    }
    if (vkBindBufferMemory(dev->device, *buf, *mem, 0) != VK_SUCCESS) {
        vkFreeMemory(dev->device, *mem, nullptr);
        vkDestroyBuffer(dev->device, *buf, nullptr);
        *buf = VK_NULL_HANDLE;
        *mem = VK_NULL_HANDLE;
        ++dev->alloc_failures;
        return fail(err, err_size, "vkBindBufferMemory failed");
    }
    ++dev->alloc_count;
    dev->alloc_bytes += req.size;
    if (dev->alloc_bytes > dev->alloc_peak_bytes) {
        dev->alloc_peak_bytes = dev->alloc_bytes;
    }
    buf_reg_add(*buf, bytes);
    return 0;
}

void qrt_vk_free(qrt_vk_dev * dev, VkBuffer buf, VkDeviceMemory mem) {
    if (dev == NULL) {
        return;
    }
    buf_reg_del(buf);
    if (mem != VK_NULL_HANDLE) {
        vkFreeMemory(dev->device, mem, nullptr);
    }
    if (buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(dev->device, buf, nullptr);
    }
}

struct qrt_vk_cmd {
    qrt_vk_dev *  dev;
    uint32_t      rings;
    VkCommandPool pool;
    VkCommandBuffer * cmd;
    VkFence *     fence;
    int *         pending;
    int *         recording;    // a begin without a submit: recoverable on the next begin
    uint32_t      cur;
    uint32_t      dispatches;
};

int qrt_vk_cmd_create(qrt_vk_dev * dev, uint32_t rings, qrt_vk_cmd ** out,
                      char * err, size_t err_size) {
    *out = NULL;
    if (dev == NULL || rings == 0) {
        return fail(err, err_size, "a pass needs a device and at least one ring");
    }
    qrt_vk_cmd * c = (qrt_vk_cmd *) calloc(1, sizeof(qrt_vk_cmd));
    if (c == NULL) {
        return fail(err, err_size, "out of memory for the pass");
    }
    c->dev   = dev;
    c->rings = rings;
    c->cmd       = (VkCommandBuffer *) calloc(rings, sizeof(VkCommandBuffer));
    c->fence     = (VkFence *) calloc(rings, sizeof(VkFence));
    c->pending   = (int *) calloc(rings, sizeof(int));
    c->recording = (int *) calloc(rings, sizeof(int));
    if (c->cmd == NULL || c->fence == NULL || c->pending == NULL || c->recording == NULL) {
        qrt_vk_cmd_destroy(dev, c);
        return fail(err, err_size, "out of memory for the pass rings");
    }
    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dev->compute_family;
    if (vkCreateCommandPool(dev->device, &pci, nullptr, &c->pool) != VK_SUCCESS) {
        qrt_vk_cmd_destroy(dev, c);
        return fail(err, err_size, "vkCreateCommandPool (pass) failed");
    }
    VkCommandBufferAllocateInfo cbai;
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = c->pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = rings;
    VkResult r = vkAllocateCommandBuffers(dev->device, &cbai, c->cmd);
    if (r != VK_SUCCESS) {
        qrt_vk_cmd_destroy(dev, c);
        return fail(err, err_size, "vkAllocateCommandBuffers (pass) failed: VkResult %d",
                    (int) r);
    }
    for (uint32_t i = 0; i < rings; ++i) {
        VkFenceCreateInfo fci;
        memset(&fci, 0, sizeof(fci));
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(dev->device, &fci, nullptr, &c->fence[i]) != VK_SUCCESS) {
            qrt_vk_cmd_destroy(dev, c);
            return fail(err, err_size, "vkCreateFence (pass) failed");
        }
    }
    *out = c;
    return 0;
}

void qrt_vk_cmd_destroy(qrt_vk_dev * dev, qrt_vk_cmd * c) {
    if (c == NULL) {
        return;
    }
    if (dev != NULL && dev->device != VK_NULL_HANDLE) {
        if (c->cmd != NULL && c->fence != NULL && c->pending != NULL) {
            for (uint32_t i = 0; i < c->rings; ++i) {
                if (c->pending[i]) {
                    char err[128];
                    qrt_vk_cmd_wait(c, i, err, sizeof(err));
                }
                if (c->fence[i] != VK_NULL_HANDLE) {
                    vkDestroyFence(dev->device, c->fence[i], nullptr);
                }
            }
        }
        if (c->pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev->device, c->pool, nullptr);
        }
    }
    free(c->cmd);
    free(c->fence);
    free(c->pending);
    free(c->recording);
    free(c);
}

uint32_t qrt_vk_cmd_rings(const qrt_vk_cmd * c) { return c != NULL ? c->rings : 0; }

int qrt_vk_cmd_wait(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u to wait for", ring);
    }
    if (!c->pending[ring]) {
        return 0;
    }
    c->pending[ring] = 0;
    if (vkWaitForFences(c->dev->device, 1, &c->fence[ring], VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        ++c->dev->io_failures;
        return fail(err, err_size, "vkWaitForFences (pass) failed");
    }
    return 0;
}

int qrt_vk_cmd_begin(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u to begin", ring);
    }
    if (qrt_vk_cmd_wait(c, ring, err, err_size) != 0) {
        return -1;
    }
    if (c->recording[ring]) {
        // A failed record left the buffer open; close it so the ring stays usable.
        vkEndCommandBuffer(c->cmd[ring]);
        c->recording[ring] = 0;
    }
    if (vkResetFences(c->dev->device, 1, &c->fence[ring]) != VK_SUCCESS) {
        return fail(err, err_size, "vkResetFences (pass) failed");
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult r = vkBeginCommandBuffer(c->cmd[ring], &bi);
    if (r != VK_SUCCESS) {
        return fail(err, err_size, "vkBeginCommandBuffer (pass) failed: VkResult %d", (int) r);
    }
    c->cur = ring;
    c->recording[ring] = 1;
    return 0;
}

VkCommandBuffer qrt_vk_cmd_buffer(qrt_vk_cmd * c, uint32_t ring) {
    if (c == NULL || ring >= c->rings) {
        return VK_NULL_HANDLE;
    }
    return c->cmd[ring];
}

void qrt_vk_cmd_record_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
}

int qrt_vk_cmd_barrier(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u for a barrier", ring);
    }
    qrt_vk_cmd_record_barrier(c->cmd[ring]);
    return 0;
}

int qrt_vk_cmd_barrier_transfer(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u for a barrier", ring);
    }
    qrt_vk_cmd_record_barrier_transfer(c->cmd[ring]);
    return 0;
}

void qrt_vk_cmd_record_barrier_transfer(VkCommandBuffer cmd) {
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
}

void qrt_vk_cmd_count(qrt_vk_cmd * c, uint32_t dispatches) {
    if (c != NULL) {
        c->dispatches += dispatches;
    }
}

uint32_t qrt_vk_cmd_dispatches(const qrt_vk_cmd * c) {
    return c != NULL ? c->dispatches : 0;
}

int qrt_vk_cmd_submit(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size) {
    if (c == NULL || ring >= c->rings) {
        return fail(err, err_size, "no pass ring %u to submit", ring);
    }
    const VkResult r = vkEndCommandBuffer(c->cmd[ring]);
    if (r != VK_SUCCESS) {
        return fail(err, err_size, "vkEndCommandBuffer (pass) failed: VkResult %d", (int) r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &c->cmd[ring];
    if (vkQueueSubmit(c->dev->compute_queue, 1, &si, c->fence[ring]) != VK_SUCCESS) {
        ++c->dev->io_failures;
        return fail(err, err_size, "vkQueueSubmit (pass) failed");
    }
    c->pending[ring] = 1;
    c->recording[ring] = 0;
    return 0;
}

struct qrt_vk_staging {
    VkBuffer        buffer;
    VkDeviceMemory  mem;
    void *          map;
    uint64_t        bytes;       // capacity asked for
    int             coherent;
    VkCommandPool   pool;
    VkCommandBuffer cmd;
    VkFence         fence;
    int             pending;     // a copy was submitted and not waited yet
    char            label[96];   // memory type and heap, for the io check
};

const char * qrt_vk_staging_label(const qrt_vk_staging * stg) {
    return stg != NULL ? stg->label : "";
}

void qrt_vk_staging_destroy(qrt_vk_dev * dev, qrt_vk_staging * stg) {
    if (stg == NULL) {
        return;
    }
    if (dev != NULL && dev->device != VK_NULL_HANDLE) {
        if (stg->pending) {
            char err[128];
            qrt_vk_staging_wait(dev, stg, err, sizeof(err));
        }
        if (stg->fence != VK_NULL_HANDLE) {
            vkDestroyFence(dev->device, stg->fence, nullptr);
        }
        // The pool owns the command buffer, so it is never freed on its own.
        if (stg->pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev->device, stg->pool, nullptr);
        }
        if (stg->map != NULL && stg->mem != VK_NULL_HANDLE) {
            vkUnmapMemory(dev->device, stg->mem);
        }
    }
    if (stg->mem != VK_NULL_HANDLE) {
        qrt_vk_free(dev, stg->buffer, stg->mem);
    }
    delete stg;
}

uint64_t qrt_vk_staging_bytes(const qrt_vk_staging * stg) {
    return stg != NULL ? stg->bytes : 0;
}

static int staging_fail(qrt_vk_dev * dev, char * err, size_t err_size, const char * what, VkResult r) {
    ++dev->io_failures;
    return fail(err, err_size, "%s failed: VkResult %d", what, (int) r);
}

int qrt_vk_staging_create(qrt_vk_dev * dev, uint64_t bytes, qrt_vk_staging ** out,
                          char * err, size_t err_size) {
    *out = NULL;
    if (bytes == 0) {
        return fail(err, err_size, "staging buffer of 0 bytes");
    }
    qrt_vk_staging * stg = new qrt_vk_staging();
    memset(stg, 0, sizeof(*stg));
    stg->bytes = bytes;

    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = bytes;
    bci.usage       = QRT_VK_USAGE_TRANSFER_SRC | QRT_VK_USAGE_TRANSFER_DST;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(dev->device, &bci, nullptr, &stg->buffer);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkCreateBuffer (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev->device, stg->buffer, &req);
    const uint32_t type = pick_host_type(dev, req.memoryTypeBits, &stg->coherent);
    if (type == UINT32_MAX) {
        fail(err, err_size, "no host-visible memory type for a %llu MiB staging buffer",
             (unsigned long long) (bytes / MIB));
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = type;
    r = vkAllocateMemory(dev->device, &mai, nullptr, &stg->mem);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkAllocateMemory (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    r = vkBindBufferMemory(dev->device, stg->buffer, stg->mem, 0);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkBindBufferMemory (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    r = vkMapMemory(dev->device, stg->mem, 0, VK_WHOLE_SIZE, 0, &stg->map);
    if (r != VK_SUCCESS || stg->map == NULL) {
        stg->map = NULL;
        staging_fail(dev, err, err_size, "vkMapMemory (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    {
        const VkMemoryType * mt = &dev->mem.memoryTypes[type];
        uint32_t heap = 0;
        for (uint32_t h = 0; h < dev->mem.memoryHeapCount; ++h) {
            if (mt->heapIndex == h) {
                heap = h;
            }
        }
        snprintf(stg->label, sizeof(stg->label), "type=%u heap=%u host=%d coherent=%d device_local=%d",
                 type, heap, 1, stg->coherent,
                 (mt->propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 1 : 0);
    }

    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dev->transfer_family;
    r = vkCreateCommandPool(dev->device, &pci, nullptr, &stg->pool);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkCreateCommandPool (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    VkCommandBufferAllocateInfo cai;
    memset(&cai, 0, sizeof(cai));
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = stg->pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(dev->device, &cai, &stg->cmd);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkAllocateCommandBuffers (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(dev->device, &fci, nullptr, &stg->fence);
    if (r != VK_SUCCESS) {
        staging_fail(dev, err, err_size, "vkCreateFence (staging)", r);
        qrt_vk_staging_destroy(dev, stg);
        return -1;
    }
    *out = stg;
    return 0;
}

// One copy between the mapped staging bytes and a device buffer, then a fence
// wait. Callers own the host-side memcpy and the cache maintenance.
static int staging_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer buf, uint64_t offset,
                          uint64_t bytes, int to_device, char * err, size_t err_size) {
    if (stg->map == NULL) {
        ++dev->io_failures;
        return fail(err, err_size, "staging buffer is not mapped");
    }
    if (bytes > stg->bytes) {
        ++dev->io_failures;
        return fail(err, err_size, "copy of %llu bytes does not fit the %llu byte staging buffer",
                    (unsigned long long) bytes, (unsigned long long) stg->bytes);
    }
    if (buf_reg_check(buf, offset, bytes, err, err_size) != 0) {
        ++dev->io_failures;
        return -1;
    }
    if (stg->pending) {
        ++dev->io_failures;
        return fail(err, err_size, "a copy is already in flight on this staging buffer");
    }
    VkResult r = vkResetFences(dev->device, 1, &stg->fence);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkResetFences", r);
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(stg->cmd, &bi);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkBeginCommandBuffer", r);
    }
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.size      = bytes;
    region.srcOffset = to_device ? 0 : offset;
    region.dstOffset = to_device ? offset : 0;
    vkCmdCopyBuffer(stg->cmd, to_device ? stg->buffer : buf, to_device ? buf : stg->buffer,
                    1, &region);
    r = vkEndCommandBuffer(stg->cmd);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkEndCommandBuffer", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &stg->cmd;
    r = vkQueueSubmit(dev->transfer_queue, 1, &si, stg->fence);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkQueueSubmit", r);
    }
    stg->pending = 1;
    return 0;
}

int qrt_vk_staging_wait(qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t err_size) {
    if (stg == NULL || !stg->pending) {
        return 0;
    }
    stg->pending = 0;
    const VkResult r = vkWaitForFences(dev->device, 1, &stg->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkWaitForFences", r);
    }
    return 0;
}

static int staging_copy(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer buf, uint64_t offset,
                        uint64_t bytes, int to_device, char * err, size_t err_size) {
    if (staging_submit(dev, stg, buf, offset, bytes, to_device, err, err_size) != 0) {
        return -1;
    }
    return qrt_vk_staging_wait(dev, stg, err, err_size);
}

int qrt_vk_upload(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                  uint64_t dst_offset, const void * src, uint64_t bytes,
                  char * err, size_t err_size) {
    if (stg == NULL || src == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "upload needs a staging buffer, a source and a size");
    }
    if (bytes > stg->bytes) {
        ++dev->io_failures;
        return fail(err, err_size, "upload of %llu bytes does not fit the %llu byte staging buffer",
                    (unsigned long long) bytes, (unsigned long long) stg->bytes);
    }
    memcpy(stg->map, src, (size_t) bytes);
    return qrt_vk_upload_mapped(dev, stg, dst, dst_offset, bytes, err, err_size);
}

int qrt_vk_download(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                    uint64_t src_offset, void * dst, uint64_t bytes,
                    char * err, size_t err_size) {
    if (stg == NULL || dst == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "download needs a staging buffer, a destination and a size");
    }
    if (bytes > stg->bytes) {
        ++dev->io_failures;
        return fail(err, err_size, "download of %llu bytes does not fit the %llu byte staging buffer",
                    (unsigned long long) bytes, (unsigned long long) stg->bytes);
    }
    if (qrt_vk_download_mapped(dev, stg, src, src_offset, bytes, err, err_size) != 0) {
        return -1;
    }
    memcpy(dst, stg->map, (size_t) bytes);
    return 0;
}

void * qrt_vk_staging_map(qrt_vk_staging * stg) {
    return stg != NULL ? stg->map : NULL;
}

int qrt_vk_upload_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size) {
    if (stg == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "upload needs a staging buffer and a size");
    }
    if (!stg->coherent) {
        VkMappedMemoryRange range;
        memset(&range, 0, sizeof(range));
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = stg->mem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        const VkResult fr = vkFlushMappedMemoryRanges(dev->device, 1, &range);
        if (fr != VK_SUCCESS) {
            return staging_fail(dev, err, err_size, "vkFlushMappedMemoryRanges", fr);
        }
    }
    return staging_copy(dev, stg, dst, dst_offset, bytes, 1, err, err_size);
}

int qrt_vk_upload_regions(qrt_vk_dev * dev, qrt_vk_staging * stg,
                          const qrt_vk_copy_region * regions, uint32_t count,
                          char * err, size_t err_size) {
    if (stg == NULL || regions == NULL || count == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "region upload needs a staging buffer and regions");
    }
    if (stg->map == NULL || stg->pending) {
        ++dev->io_failures;
        return fail(err, err_size, "the staging buffer is not free for a region upload");
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (regions[i].src == NULL || regions[i].bytes == 0) {
            ++dev->io_failures;
            return fail(err, err_size, "region %u has no source or size", i);
        }
        if (regions[i].bytes > stg->bytes - total ||
            buf_reg_check(regions[i].dst, regions[i].dst_offset, regions[i].bytes, err,
                          err_size) != 0) {
            ++dev->io_failures;
            return -1;
        }
        total += regions[i].bytes;
    }
    // Pack the sources only when they are not already in packed order: the cache path
    // reads experts straight to their packed offsets, so this is the common case.
    {
        const unsigned char * base = (const unsigned char *) stg->map;
        uint64_t packed = 0;
        int in_place = 1;
        for (uint32_t i = 0; i < count; ++i) {
            if (regions[i].src != base + packed) {
                in_place = 0;
                break;
            }
            packed += regions[i].bytes;
        }
        if (!in_place) {
            std::vector<unsigned char> tmp((size_t) total);
            packed = 0;
            for (uint32_t i = 0; i < count; ++i) {
                memcpy(tmp.data() + packed, regions[i].src, (size_t) regions[i].bytes);
                packed += regions[i].bytes;
            }
            memcpy(stg->map, tmp.data(), (size_t) total);
        }
    }
    VkResult r = vkResetFences(dev->device, 1, &stg->fence);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkResetFences", r);
    }
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(stg->cmd, &bi);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkBeginCommandBuffer", r);
    }
    uint64_t src_off = 0;
    for (uint32_t i = 0; i < count; ++i) {
        VkBufferCopy region;
        memset(&region, 0, sizeof(region));
        region.srcOffset = src_off;
        region.dstOffset = regions[i].dst_offset;
        region.size      = regions[i].bytes;
        vkCmdCopyBuffer(stg->cmd, stg->buffer, regions[i].dst, 1, &region);
        src_off += regions[i].bytes;
    }
    r = vkEndCommandBuffer(stg->cmd);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkEndCommandBuffer", r);
    }
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &stg->cmd;
    r = vkQueueSubmit(dev->transfer_queue, 1, &si, stg->fence);
    if (r != VK_SUCCESS) {
        return staging_fail(dev, err, err_size, "vkQueueSubmit", r);
    }
    stg->pending = 1;
    return qrt_vk_staging_wait(dev, stg, err, err_size);
}

int qrt_vk_upload_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size) {
    if (stg == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "upload needs a staging buffer and a size");
    }
    if (!stg->coherent) {
        VkMappedMemoryRange range;
        memset(&range, 0, sizeof(range));
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = stg->mem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        const VkResult fr = vkFlushMappedMemoryRanges(dev->device, 1, &range);
        if (fr != VK_SUCCESS) {
            return staging_fail(dev, err, err_size, "vkFlushMappedMemoryRanges", fr);
        }
    }
    return staging_submit(dev, stg, dst, dst_offset, bytes, 1, err, err_size);
}

int qrt_vk_download_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                           uint64_t src_offset, uint64_t bytes, char * err, size_t err_size) {
    if (stg == NULL || bytes == 0) {
        ++dev->io_failures;
        return fail(err, err_size, "download needs a staging buffer and a size");
    }
    if (staging_copy(dev, stg, src, src_offset, bytes, 0, err, err_size) != 0) {
        return -1;
    }
    if (!stg->coherent) {
        VkMappedMemoryRange range;
        memset(&range, 0, sizeof(range));
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = stg->mem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        const VkResult ir = vkInvalidateMappedMemoryRanges(dev->device, 1, &range);
        if (ir != VK_SUCCESS) {
            return staging_fail(dev, err, err_size, "vkInvalidateMappedMemoryRanges", ir);
        }
    }
    return 0;
}

void qrt_vk_print_info(const qrt_vk_dev * dev, FILE * f) {
    fprintf(f, "device=%s\n", dev->name);
    fprintf(f, "heap_mib=%llu budget_mib=%llu usage_mib=%llu tier_mib=%u memory_budget_ext=%d\n",
            (unsigned long long) (dev->heap_bytes / MIB),
            (unsigned long long) (dev->budget_bytes / MIB),
            (unsigned long long) (dev->usage_bytes / MIB), dev->tier_mib,
            dev->has_memory_budget);
    fprintf(f, "queue_compute=%u queue_transfer=%u timestamp_bits=%u\n",
            dev->compute_family, dev->transfer_family, dev->timestamp_bits);
    fprintf(f, "alloc_count=%u alloc_mib=%llu alloc_peak_mib=%llu alloc_failures=%u "
               "cache_reductions=%u io_failures=%u\n",
            dev->alloc_count, (unsigned long long) (dev->alloc_bytes / MIB),
            (unsigned long long) (dev->alloc_peak_bytes / MIB),
            dev->alloc_failures, dev->cache_reductions, dev->io_failures);
    fprintf(f, "subgroup_size=%u subgroup_arithmetic=%d dot_product=%d\n", dev->subgroup_size,
            dev->has_subgroup_arithmetic, dev->has_dot_product);
}

#else // QRT_HAVE_VULKAN

int qrt_vk_open(uint32_t tier_mib, qrt_vk_dev ** out, char * err, size_t err_size) {
    (void) tier_mib;
    *out = NULL;
    snprintf(err, err_size, "this build has no Vulkan (configure with the Vulkan SDK)");
    return -1;
}

void qrt_vk_close(qrt_vk_dev * dev) {
    delete dev;
}

int qrt_vk_query_budget(qrt_vk_dev * dev, char * err, size_t err_size) {
    (void) dev;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_plan_budget(qrt_vk_dev * dev, const qrt_config * cfg, uint64_t expert_store_bytes,
                qrt_vk_plan * out, char * err, size_t err_size) {
    (void) dev;
    (void) cfg;
    (void) expert_store_bytes;
    memset(out, 0, sizeof(*out));
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_alloc(qrt_vk_dev * dev, uint64_t bytes, VkBufferUsageFlags usage,
                 VkBuffer * buf, VkDeviceMemory * mem, char * err, size_t err_size) {
    (void) dev;
    (void) bytes;
    (void) usage;
    *buf = NULL;
    *mem = NULL;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

void qrt_vk_free(qrt_vk_dev * dev, VkBuffer buf, VkDeviceMemory mem) {
    (void) dev;
    (void) buf;
    (void) mem;
}

struct qrt_vk_staging;

int qrt_vk_staging_create(qrt_vk_dev * dev, uint64_t bytes, qrt_vk_staging ** out,
                          char * err, size_t err_size) {
    (void) dev;
    (void) bytes;
    *out = NULL;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

void qrt_vk_staging_destroy(qrt_vk_dev * dev, qrt_vk_staging * stg) {
    (void) dev;
    (void) stg;
}

uint64_t qrt_vk_staging_bytes(const qrt_vk_staging * stg) {
    (void) stg;
    return 0;
}

const char * qrt_vk_staging_label(const qrt_vk_staging * stg) {
    (void) stg;
    return "";
}

void * qrt_vk_staging_map(qrt_vk_staging * stg) {
    (void) stg;
    return NULL;
}

int qrt_vk_upload_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) dst;
    (void) dst_offset;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_download_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                           uint64_t src_offset, uint64_t bytes, char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) src;
    (void) src_offset;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_upload_regions(qrt_vk_dev * dev, qrt_vk_staging * stg,
                          const qrt_vk_copy_region * regions, uint32_t count,
                          char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) regions;
    (void) count;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_upload_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) dst;
    (void) dst_offset;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_staging_wait(qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_upload(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                  uint64_t dst_offset, const void * src, uint64_t bytes,
                  char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) dst;
    (void) dst_offset;
    (void) src;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

int qrt_vk_download(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                    uint64_t src_offset, void * dst, uint64_t bytes,
                    char * err, size_t err_size) {
    (void) dev;
    (void) stg;
    (void) src;
    (void) src_offset;
    (void) dst;
    (void) bytes;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

void qrt_vk_print_info(const qrt_vk_dev * dev, FILE * f) {
    (void) dev;
    fprintf(f, "device=none (built without Vulkan)\n");
}

struct qrt_vk_cmd { int unused; };
int qrt_vk_cmd_create(qrt_vk_dev *, uint32_t, qrt_vk_cmd ** out, char * err, size_t err_size) {
    *out = NULL;
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
void qrt_vk_cmd_destroy(qrt_vk_dev *, qrt_vk_cmd *) {}
uint32_t qrt_vk_cmd_rings(const qrt_vk_cmd *) { return 0; }
int qrt_vk_cmd_begin(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
VkCommandBuffer qrt_vk_cmd_buffer(qrt_vk_cmd *, uint32_t) { return VK_NULL_HANDLE; }
int qrt_vk_cmd_barrier(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
int qrt_vk_cmd_barrier_transfer(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
void qrt_vk_cmd_record_barrier_transfer(VkCommandBuffer) {}
void qrt_vk_cmd_record_barrier(VkCommandBuffer) {}
void qrt_vk_cmd_count(qrt_vk_cmd *, uint32_t) {}
uint32_t qrt_vk_cmd_dispatches(const qrt_vk_cmd *) { return 0; }
int qrt_vk_cmd_submit(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}
int qrt_vk_cmd_wait(qrt_vk_cmd *, uint32_t, char * err, size_t err_size) {
    snprintf(err, err_size, "this build has no Vulkan");
    return -1;
}

#endif // QRT_HAVE_VULKAN
