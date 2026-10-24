#include "vk_dev.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef QRT_HAVE_VULKAN

namespace {

const uint64_t MIB = 1024ull * 1024ull;

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

    VkPhysicalDeviceVulkan12Features f12;
    memset(&f12, 0, sizeof(f12));
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
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
    VkPhysicalDeviceVulkan12Features enable12;
    memset(&enable12, 0, sizeof(enable12));
    enable12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enable12.timelineSemaphore = VK_TRUE;

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

    const char * ext_name = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext                   = &enable12;
    dci.queueCreateInfoCount    = n_qci;
    dci.pQueueCreateInfos       = qci;
    dci.enabledExtensionCount   = dev->has_memory_budget ? 1u : 0u;
    dci.ppEnabledExtensionNames = &ext_name;
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

int qrt_vk_plan_budget(qrt_vk_dev * dev, const qrt_config * cfg, uint64_t expert_store_bytes,
                qrt_vk_plan * out, char * err, size_t err_size) {
    memset(out, 0, sizeof(*out));
    const uint64_t kv_per_token = cfg->kv_precision == QRT_KV_F16 ? 22528ull : 6336ull;
    const uint64_t kv_bytes = (uint64_t) cfg->kv_page_tokens * kv_per_token;
    out->dense_bytes   = DENSE_BYTES;
    out->kv_bytes      = kv_bytes;
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
    return 0;
}

void qrt_vk_free(qrt_vk_dev * dev, VkBuffer buf, VkDeviceMemory mem) {
    if (dev == NULL) {
        return;
    }
    if (mem != VK_NULL_HANDLE) {
        vkFreeMemory(dev->device, mem, nullptr);
    }
    if (buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(dev->device, buf, nullptr);
    }
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
    fprintf(f, "alloc_count=%u alloc_mib=%llu alloc_failures=%u cache_reductions=%u\n",
            dev->alloc_count, (unsigned long long) (dev->alloc_bytes / MIB),
            dev->alloc_failures, dev->cache_reductions);
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

void qrt_vk_print_info(const qrt_vk_dev * dev, FILE * f) {
    (void) dev;
    fprintf(f, "device=none (built without Vulkan)\n");
}

#endif // QRT_HAVE_VULKAN
