// vkpage.cpp - M0 probes for the paged-KV plan (bench/THREE-TIER.md F10).
//
// table: M0c. Logical KV cells are mapped onto blocks that live in different VkBuffers,
//        in a shuffled order. A table of 64-bit device addresses is read by the shader;
//        the write pass goes through the table as well. The host then verifies that every
//        logical cell landed in the right physical block and that no other byte moved.
// bench: M0d. The decode-style FA read loop (2 KiB per cell: K row + V row, 512 f16 each)
//        once over a single contiguous buffer and once through the block table. Same bytes,
//        same instruction mix; only the address computation differs.
//
// Build (MinGW, Vulkan SDK), from bench/tools:
//   glslc -O vkpage_rw.comp -o vkpage_rw.spv
//   glslc -O vkpage_fa.comp -o vkpage_fa.spv
//   glslc -O -DPAGED=1 vkpage_fa.comp -o vkpage_fa_paged.spv
//   g++ -O2 -std=c++17 vkpage.cpp -I"%VULKAN_SDK%\Include" -L"%VULKAN_SDK%\Lib" -lvulkan-1 -o vkpage.exe
//
// Usage: vkpage.exe info | table | bench [cells] [iters]

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
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

static std::vector<char> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::printf("cannot open %s\n", path.c_str());
        std::exit(1);
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data((size_t) size);
    f.read(data.data(), size);
    return data;
}

struct Buf {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
    void *         map = nullptr;
    uint64_t       addr = 0;
};

struct App {
    VkInstance       inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties       props{};
    VkPhysicalDeviceMemoryProperties memprops{};
    VkDevice         dev = VK_NULL_HANDLE;
    uint32_t         fam = 0;
    VkQueue          q = VK_NULL_HANDLE;
    VkCommandPool    pool = VK_NULL_HANDLE;
};

static uint32_t mem_type(App & a, uint32_t bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < a.memprops.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (a.memprops.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    std::printf("no memory type for bits=0x%x props=0x%x\n", bits, (unsigned) props);
    std::exit(1);
}

static Buf make_buf(App & a, VkDeviceSize size, bool host_visible) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(a.dev, &bci, nullptr, &b.buf));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(a.dev, b.buf, &req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext           = &flags;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = mem_type(a, req.memoryTypeBits,
            host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                         : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(a.dev, &mai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(a.dev, b.buf, b.mem, 0));
    if (host_visible) {
        VK_CHECK(vkMapMemory(a.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    }

    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdai.buffer = b.buf;
    b.addr = vkGetBufferDeviceAddress(a.dev, &bdai);
    return b;
}

static void destroy_buf(App & a, Buf & b) {
    if (b.buf) vkDestroyBuffer(a.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(a.dev, b.mem, nullptr);
    b = Buf{};
}

static VkPipeline make_pipeline(App & a, VkPipelineLayout pl, const std::string & spv) {
    const std::vector<char> code = read_file(spv);
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
    cpci.layout       = pl;
    VkPipeline p = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(a.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
    vkDestroyShaderModule(a.dev, mod, nullptr);
    return p;
}

static VkCommandBuffer begin_cb(App & a) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = a.pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(a.dev, &cai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

static void submit_cb(App & a, VkCommandBuffer cb) {
    VK_CHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(a.dev, &fci, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VK_CHECK(vkQueueSubmit(a.q, 1, &si, fence));
    VK_CHECK(vkWaitForFences(a.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(a.dev, fence, nullptr);
    vkFreeCommandBuffers(a.dev, a.pool, 1, &cb);
}

static void init(App & a, int device_index) {
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
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        std::printf("  [%u] %s\n", i, p.deviceName);
    }
    // the model runs on -dev Vulkan1, i.e. the second enumerated device on this machine
    int pick = device_index;
    if (pick < 0 || pick >= (int) n) {
        pick = n > 1 ? 1 : 0;
    }
    a.phys = devs[pick];
    vkGetPhysicalDeviceProperties(a.phys, &a.props);
    vkGetPhysicalDeviceMemoryProperties(a.phys, &a.memprops);
    std::printf("picked [%d]: %s\n", pick, a.props.deviceName);

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDevice16BitStorageFeatures f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    f12.pNext = &f16;
    vkGetPhysicalDeviceFeatures2(a.phys, &f2);
    std::printf("bufferDeviceAddress=%d shaderInt64=%d storageBuffer16BitAccess=%d\n",
            (int) f12.bufferDeviceAddress, (int) f2.features.shaderInt64, (int) f16.storageBuffer16BitAccess);
    if (!f12.bufferDeviceAddress || !f2.features.shaderInt64 || !f16.storageBuffer16BitAccess) {
        std::printf("the paged-KV plan needs all three of the features above\n");
        std::exit(1);
    }

    uint32_t nf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(a.phys, &nf, nullptr);
    std::vector<VkQueueFamilyProperties> fams(nf);
    vkGetPhysicalDeviceQueueFamilyProperties(a.phys, &nf, fams.data());
    int compute_fam = -1;
    for (uint32_t i = 0; i < nf; ++i) {
        const bool has_c = (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool has_g = (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (has_c && !has_g) {
            compute_fam = (int) i;
            break;
        }
        if (has_c && compute_fam < 0) {
            compute_fam = (int) i;
        }
    }
    if (compute_fam < 0) {
        std::printf("no compute queue family\n");
        std::exit(1);
    }
    a.fam = (uint32_t) compute_fam;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = a.fam;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    e12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDevice16BitStorageFeatures e16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    e16.storageBuffer16BitAccess = VK_TRUE;
    e16.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    e16.pNext = nullptr;
    e12.pNext = &e16;

    VkPhysicalDeviceFeatures core{};
    core.shaderInt64 = VK_TRUE;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext                = &e12;
    dci.pEnabledFeatures     = &core;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    VK_CHECK(vkCreateDevice(a.phys, &dci, nullptr, &a.dev));
    vkGetDeviceQueue(a.dev, a.fam, 0, &a.q);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = a.fam;
    VK_CHECK(vkCreateCommandPool(a.dev, &cpci, nullptr, &a.pool));
}

// small exact integers only, so a plain float compare is enough
static float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        bits = sign;  // only zeros are written by the probe
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

struct CellVal {
    float c, d, b, one;
};

static CellVal expected_val(uint32_t c, uint32_t d, uint32_t bs) {
    return CellVal{ (float) (c % 512u), (float) d, (float) (c / bs), 1.0f };
}

static bool check_vec(const uint16_t * v, uint32_t c, uint32_t d, uint32_t bs) {
    const CellVal e = expected_val(c, d, bs);
    const float got[4] = { half_to_float(v[0]), half_to_float(v[1]), half_to_float(v[2]), half_to_float(v[3]) };
    const float want[4] = { e.c, e.d, e.b, e.one };
    for (int i = 0; i < 4; ++i) {
        if (got[i] != want[i]) {
            std::printf("mismatch cell=%u d=%u comp=%d got=%f want=%f\n", c, d, i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

struct PcRW {
    uint64_t table;
    uint64_t out_ref;
    uint32_t n_cells;
    uint32_t bs;
    uint32_t cell4;
    uint32_t mode;
};

struct PcFA {
    uint64_t k_base;
    uint64_t v_base;
    uint64_t table;
    uint64_t out_ref;
    uint32_t n_kv;
    uint32_t bs;
    uint32_t cell4;
    uint32_t n_blocks;
    uint32_t chunk;
    uint32_t pad = 0;
};

// logical block -> (buffer, slot), deliberately not the identity
static void block_place(uint32_t l, uint32_t n_bufs, uint32_t & buf, uint32_t & slot) {
    buf  = (l * 2 + 1) % n_bufs;
    slot = l / n_bufs;
}

static int run_table(App & a) {
    const uint32_t bs      = 64;
    const uint32_t cell4   = 4;
    const uint32_t n_blocks = 8;
    const uint32_t n_cells = bs * n_blocks;
    const uint32_t n_bufs  = 3;
    const uint32_t per_buf = (n_blocks + n_bufs - 1) / n_bufs;

    const VkDeviceSize blk_bytes = (VkDeviceSize) bs * cell4 * 8;
    const VkDeviceSize buf_bytes = blk_bytes * per_buf;

    std::vector<Buf> blocks;
    for (uint32_t i = 0; i < n_bufs; ++i) {
        blocks.push_back(make_buf(a, buf_bytes, true));
        memset(blocks.back().map, 0, (size_t) buf_bytes);
    }

    Buf table = make_buf(a, (VkDeviceSize) n_blocks * 8, true);
    uint64_t * tab = (uint64_t *) table.map;
    for (uint32_t l = 0; l < n_blocks; ++l) {
        uint32_t bi = 0, slot = 0;
        block_place(l, n_bufs, bi, slot);
        tab[l] = blocks[bi].addr + (uint64_t) slot * blk_bytes;
        std::printf("  logical block %u -> buffer %u slot %u\n", l, bi, slot);
    }

    Buf out = make_buf(a, (VkDeviceSize) n_cells * cell4 * 8, true);
    memset(out.map, 0, (size_t) out.size);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(PcRW);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(a.dev, &plci, nullptr, &pl));

    const std::string dir = std::string(".");
    VkPipeline pipe = make_pipeline(a, pl, dir + "/vkpage_rw.spv");

    const uint32_t groups = 16;
    PcRW pc{};
    pc.table   = table.addr;
    pc.n_cells = n_cells;
    pc.bs      = bs;
    pc.cell4   = cell4;

    for (uint32_t mode = 0; mode < 2; ++mode) {
        pc.mode    = mode;
        pc.out_ref = mode == 0 ? 0 : out.addr;

        VkCommandBuffer cb = begin_cb(a);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, groups, 1, 1);
        submit_cb(a, cb);
    }

    bool ok = true;
    // every logical cell must be readable through the table, and the physical layout must match
    for (uint32_t c = 0; c < n_cells; ++c) {
        for (uint32_t d = 0; d < cell4; ++d) {
            if (!check_vec((const uint16_t *) out.map + (c * cell4 + d) * 4, c, d, bs)) {
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }
    std::printf("  read-through-table: %s\n", ok ? "ok" : "FAIL");

    // every byte outside the logical blocks must still be zero
    uint32_t stray = 0;
    {
        std::vector<std::vector<char>> used(n_bufs, std::vector<char>(per_buf, 0));
        for (uint32_t l = 0; l < n_blocks; ++l) {
            uint32_t bi = 0, slot = 0;
            block_place(l, n_bufs, bi, slot);
            used[bi][slot] = 1;
        }
        const size_t halves_per_blk = (size_t) bs * cell4 * 4;
        for (uint32_t bi = 0; bi < n_bufs; ++bi) {
            const uint16_t * base = (const uint16_t *) blocks[bi].map;
            for (uint32_t slot = 0; slot < per_buf; ++slot) {
                if (used[bi][slot]) {
                    continue;
                }
                const uint16_t * p = base + (size_t) slot * halves_per_blk;
                for (size_t i = 0; i < halves_per_blk; ++i) {
                    if (p[i] != 0) {
                        ++stray;
                    }
                }
            }
        }
    }
    std::printf("  stray non-zero halves in unmapped blocks: %u\n", stray);
    if (stray != 0) ok = false;

    // the physical blocks must hold the expected values as well
    bool layout_ok = true;
    for (uint32_t l = 0; l < n_blocks && layout_ok; ++l) {
        uint32_t bi = 0, slot = 0;
        block_place(l, n_bufs, bi, slot);
        const uint16_t * base = (const uint16_t *) blocks[bi].map + (size_t) slot * bs * cell4 * 4;
        for (uint32_t r = 0; r < bs && layout_ok; ++r) {
            for (uint32_t d = 0; d < cell4; ++d) {
                const uint32_t c = l * bs + r;
                if (!check_vec(base + (r * cell4 + d) * 4, c, d, bs)) {
                    layout_ok = false;
                    break;
                }
            }
        }
    }
    std::printf("  physical layout: %s\n", layout_ok ? "ok" : "FAIL");
    if (!layout_ok) ok = false;

    vkDeviceWaitIdle(a.dev);
    vkDestroyPipeline(a.dev, pipe, nullptr);
    vkDestroyPipelineLayout(a.dev, pl, nullptr);
    destroy_buf(a, table);
    destroy_buf(a, out);
    for (auto & b : blocks) destroy_buf(a, b);

    std::printf("table test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static double time_fa(App & a, VkPipeline pipe, VkPipelineLayout pl, const PcFA & pc, uint32_t groups, uint32_t iters) {
    VkCommandBuffer cb = begin_cb(a);
    for (uint32_t i = 0; i < iters; ++i) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), (const void *) &pc);
        vkCmdDispatch(cb, groups, 1, 1);
    }
    const auto t0 = std::chrono::steady_clock::now();
    submit_cb(a, cb);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static void fill_buf(App & a, Buf & dst, Buf & staging) {
    for (VkDeviceSize off = 0; off < dst.size; off += staging.size) {
        const VkDeviceSize n = std::min<VkDeviceSize>(staging.size, dst.size - off);
        VkCommandBuffer cb = begin_cb(a);
        VkBufferCopy c{};
        c.srcOffset = 0;
        c.dstOffset = off;
        c.size      = n;
        vkCmdCopyBuffer(cb, staging.buf, dst.buf, 1, &c);
        submit_cb(a, cb);
    }
}

static void run_bench(App & a, uint32_t cells, uint32_t iters, uint32_t chunk) {
    const uint32_t bs     = 2048;
    const uint32_t cell4  = 128;   // 512 f16 per cell for K, same for V
    const uint32_t n_blk  = (cells + bs - 1) / bs;
    const uint32_t groups = (cells + chunk - 1) / chunk;

    const VkDeviceSize contig_bytes = (VkDeviceSize) cells * cell4 * 8;
    const VkDeviceSize blk_bytes    = (VkDeviceSize) bs * cell4 * 8;

    Buf staging = make_buf(a, 4u << 20, true);
    memset(staging.map, 1, (size_t) staging.size);

    Buf k_contig = make_buf(a, contig_bytes, false);
    Buf v_contig = make_buf(a, contig_bytes, false);
    fill_buf(a, k_contig, staging);
    fill_buf(a, v_contig, staging);

    std::vector<Buf> k_blk, v_blk;
    for (uint32_t i = 0; i < n_blk; ++i) {
        k_blk.push_back(make_buf(a, blk_bytes, false));
        v_blk.push_back(make_buf(a, blk_bytes, false));
        fill_buf(a, k_blk.back(), staging);
        fill_buf(a, v_blk.back(), staging);
    }

    Buf table = make_buf(a, (VkDeviceSize) 2 * n_blk * 8, true);
    uint64_t * tab = (uint64_t *) table.map;
    for (uint32_t i = 0; i < n_blk; ++i) {
        tab[i]           = k_blk[i].addr;
        tab[n_blk + i]   = v_blk[i].addr;
    }

    Buf out = make_buf(a, (VkDeviceSize) groups * 4, true);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(PcFA);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(a.dev, &plci, nullptr, &pl));

    VkPipeline pipe_contig = make_pipeline(a, pl, "./vkpage_fa.spv");
    VkPipeline pipe_paged  = make_pipeline(a, pl, "./vkpage_fa_paged.spv");

    PcFA pc{};
    pc.k_base   = k_contig.addr;
    pc.v_base   = v_contig.addr;
    pc.table    = table.addr;
    pc.out_ref  = out.addr;
    pc.n_kv     = cells;
    pc.bs       = bs;
    pc.cell4    = cell4;
    pc.n_blocks = n_blk;
    pc.chunk    = chunk;

    const double bytes = (double) cells * 2 * 1024.0;

    time_fa(a, pipe_contig, pl, pc, groups, 2);
    time_fa(a, pipe_paged,  pl, pc, groups, 2);

    const double ms_c = time_fa(a, pipe_contig, pl, pc, groups, iters);
    const double ms_p = time_fa(a, pipe_paged,  pl, pc, groups, iters);

    std::printf("cells=%u (%.1f MiB K+V) groups=%u chunk=%u iters=%u\n",
            cells, bytes / (1024.0 * 1024.0), groups, chunk, iters);
    std::printf("  contiguous: %8.3f ms  %7.1f GB/s\n", ms_c, bytes / ms_c / 1e6);
    std::printf("  paged     : %8.3f ms  %7.1f GB/s  (+%.2f%%)\n", ms_p, bytes / ms_p / 1e6,
            100.0 * (ms_p - ms_c) / ms_c);

    vkDeviceWaitIdle(a.dev);
    vkDestroyPipeline(a.dev, pipe_contig, nullptr);
    vkDestroyPipeline(a.dev, pipe_paged, nullptr);
    vkDestroyPipelineLayout(a.dev, pl, nullptr);
    destroy_buf(a, staging);
    destroy_buf(a, k_contig);
    destroy_buf(a, v_contig);
    destroy_buf(a, table);
    destroy_buf(a, out);
    for (auto & b : k_blk) destroy_buf(a, b);
    for (auto & b : v_blk) destroy_buf(a, b);
}

int main(int argc, char ** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "info";

    App a;
    const char * idx_env = getenv("VKPAGE_DEV");
    init(a, idx_env ? atoi(idx_env) : -1);

    int rc = 0;
    if (cmd == "table") {
        rc = run_table(a);
    } else if (cmd == "bench") {
        const uint32_t cells = argc > 2 ? (uint32_t) atoi(argv[2]) : 131072;
        const uint32_t iters = argc > 3 ? (uint32_t) atoi(argv[3]) : 20;
        const uint32_t chunk = argc > 4 ? (uint32_t) atoi(argv[4]) : 256;
        run_bench(a, cells, iters, chunk);
    } else {
        std::printf("usage: vkpage.exe info | table | bench [cells] [iters] [chunk]\n");
    }

    vkDeviceWaitIdle(a.dev);
    vkDestroyCommandPool(a.dev, a.pool, nullptr);
    vkDestroyDevice(a.dev, nullptr);
    vkDestroyInstance(a.inst, nullptr);
    return rc;
}
