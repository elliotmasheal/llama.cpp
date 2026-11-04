// PLAN-A upload ring: worker-thread staging for expert weight uploads.
//
// Isolated behind its own header (audit requirement): included only by
// ggml-vulkan.cpp; the ring is created only when GGML_VK_UPLOAD_THREAD=1 and
// destroyed deterministically with the backend context.
//
// Thread model: the worker performs PrefetchVirtualMemory + memcpy from the
// mmap'd source into ring slots (a sysmem non-device-local vk_buffer). It
// NEVER calls Vulkan. All Vulkan calls (copyBuffer recording, submissions,
// semaphore queries) happen on the host thread inside submit_ready() /
// submit_all() / retire_completed(), called from the caller's normal submit
// points. Slot lifecycle is monotonic:
//   begin -> (caller fills) -> seal -> ready -> submitted -> retired
// with four counters that only advance; a slot's memory is reused only after
// the GPU DMA that reads it has been observed complete (fence drain or
// timeline semaphore query), so the CPU memcpy into a slot can never race the
// DMA reading it.

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <memoryapi.h>
#endif

struct vk_upload_ring {
    static constexpr size_t SLOT_CAP = 256ull << 20;   // per-slot staging capacity
    static constexpr size_t N_SLOTS  = 8;              // 2 GiB total (sysmem)

    // one upload request: up to 4 entries (the PP path uploads at most two
    // expert range segments + padding per layer)
    static constexpr size_t MAX_ENTRIES = 4;
    struct req {
        ggml_tensor * tensors[MAX_ENTRIES];
        const void *  data[MAX_ENTRIES];
        size_t        offsets[MAX_ENTRIES];
        size_t        sizes[MAX_ENTRIES];
        size_t        n = 0;
        size_t        staging_off = 0;          // slot base inside the ring buffer
        uint64_t      sem_value_at_submit = 0;  // timeline value covering its DMA
        enum class state_t : int { free, filling, ready, submitted, retired };
        state_t       state = state_t::free;
    };

    vk_buffer buffer;                       // sysmem cached, non-device-local
    std::vector<req> slots;                 // N_SLOTS entries, indexed seq % N_SLOTS

    uint64_t push_seq   = 0;                // slots sealed to the worker
    uint64_t ready_seq  = 0;                // worker finished memcpy
    uint64_t submit_seq = 0;                // copyBuffer recorded + submitted
    uint64_t retire_seq = 0;                // DMA observed complete

    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv_work;        // wakes worker on seal / stop
    std::condition_variable cv_host;        // wakes host on ready / retired
    std::atomic<bool>       stop{false};
    bool                    ok = false;     // false -> caller uses the inline fallback

    ~vk_upload_ring() {
        stop_worker();
    }

    // create the ring memory (sysmem cached, non-device-local) and start the worker;
    // returns false (with nothing allocated) when the environment disables it, the
    // device cannot pipeline transfers, or allocation fails, so the caller keeps its
    // inline fallback
    bool init(ggml_backend_vk_context * ctx, size_t total_cap) {
        if (getenv("GGML_VK_UPLOAD_THREAD") == nullptr) {
            return false;
        }
        // without an independent transfer queue or with serialized submissions the
        // ring cannot overlap anything; keep the blocking fallback
        if (!ctx->device->async_use_transfer_queue || ctx->device->serialize_submissions) {
            return false;
        }
        // ring memory must survive the worker; created outside the garbage collector
        buffer = ggml_vk_create_sysmem_buffer(ctx->device, total_cap);
        if (buffer == nullptr) {
            return false;
        }
        slots.assign(N_SLOTS, {});
        for (size_t i = 0; i < N_SLOTS; ++i) {
            slots[i].staging_off = i * SLOT_CAP;
        }
        ok = true;
        worker = std::thread([this]() { worker_loop(); });
        ggml_perf::count("vk", "upload_ring_start", total_cap);
        return true;
    }

    void stop_worker() {
        if (!ok) {
            return;
        }
        stop.store(true, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mtx);
            cv_work.notify_all();
        }
        if (worker.joinable()) {
            worker.join();
        }
        ok = false;
    }

    void destroy() {
        stop_worker();
        if (buffer != nullptr) {
            ggml_vk_destroy_buffer(buffer);
        }
        slots.clear();
    }

    // reserve a free slot (does NOT hand it to the worker yet): blocks while no free
    // slot exists, advancing retirement via the timeline semaphore. Returns nullptr
    // when the ring is disabled or the request cannot be staged; the caller then
    // uses its blocking fallback. Fill the returned req and call seal() exactly once.
    req * begin(ggml_backend_vk_context * ctx, size_t total) {
        if (!ok || total > SLOT_CAP) {
            return nullptr;
        }
        std::unique_lock<std::mutex> lk(mtx);
        while (push_seq - retire_seq >= N_SLOTS) {
            retire_completed(ctx, &lk);
            if (push_seq - retire_seq < N_SLOTS) {
                break;
            }
            cv_host.wait_for(lk, std::chrono::milliseconds(1));
        }
        req & r = slots[push_seq % N_SLOTS];
        r.n = 0;
        r.state = req::state_t::filling;
        return &r;
    }

    // hand a previously reserved slot to the worker (worker starts memcpy)
    void seal(req * r) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            r->state = req::state_t::filling;
            push_seq++;
        }
        cv_work.notify_all();
    }

    // return a reserved-but-unused slot (nothing was handed to the worker)
    void cancel(req * r) {
        std::lock_guard<std::mutex> lk(mtx);
        r->n = 0;
        r->state = req::state_t::free;
        cv_host.notify_all();
    }

    // record copyBuffer commands for every slot the worker has finished and submit
    // them as one transfer submission; never blocks on the worker (a still-filling
    // slot is simply left for a later call). Returns the number of slots submitted.
    size_t submit_ready(ggml_backend_vk_context * ctx) {
        if (!ok || submit_seq == push_seq) {
            return 0;
        }
        uint64_t upto;
        {
            std::lock_guard<std::mutex> lk(mtx);
            upto = submit_seq;
            while (upto < push_seq && (int) slots[upto % N_SLOTS].state >= (int) req::state_t::ready) {
                upto++;
            }
        }
        if (upto == submit_seq) {
            return 0;   // nothing ready yet: keep the pipeline full, no waiting
        }
        vk_context cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        // record the copies for every ready slot (host thread; worker only memcpy'd)
        for (uint64_t s = submit_seq; s < upto; ++s) {
            req & r = slots[s % N_SLOTS];
            vk_buffer group_buf = nullptr;
            size_t entry_off = 0;
            std::vector<vk::BufferCopy> slices;
            for (size_t i = 0; i < r.n; ++i) {
                ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *) r.tensors[i]->buffer->context;
                if (buf_ctx->dev_buffer != group_buf) {
                    if (!slices.empty()) {
                        cpy_ctx->s->buffer->buf.copyBuffer(buffer->buffer, group_buf->buffer, slices);
                        slices.clear();
                    }
                    group_buf = buf_ctx->dev_buffer;
                }
                vk::BufferCopy slice;
                slice.srcOffset = r.staging_off + entry_off;
                slice.dstOffset = vk_tensor_offset(r.tensors[i]) + r.tensors[i]->view_offs + r.offsets[i];
                slice.size      = r.sizes[i];
                slices.push_back(slice);
                entry_off += r.sizes[i];
            }
            if (!slices.empty()) {
                cpy_ctx->s->buffer->buf.copyBuffer(buffer->buffer, group_buf->buffer, slices);
            }
        }
        if (!ggml_vk_submit_transfer_ctx(ctx)) {
            return 0;   // submission refused: leave state untouched, retry next call
        }
        // the submission above advanced the timeline semaphore by one; its signal
        // covers every DMA recorded here
        const uint64_t value_after = ctx->transfer_semaphore.value;
        {
            std::lock_guard<std::mutex> lk(mtx);
            const uint64_t n_submitted = upto - submit_seq;
            for (uint64_t s = submit_seq; s < upto; ++s) {
                req & r = slots[s % N_SLOTS];
                r.state = req::state_t::submitted;
                r.sem_value_at_submit = value_after;
            }
            submit_seq = upto;
            return (size_t) n_submitted;
        }
    }

    // blocking flush used at fence-drain points: wait for every pushed slot to
    // become ready, then submit all of them so the caller's subsequent fence wait
    // also covers the ring's DMAs
    void submit_all(ggml_backend_vk_context * ctx) {
        if (!ok || submit_seq == push_seq) {
            return;
        }
        {
            std::unique_lock<std::mutex> lk(mtx);
            for (uint64_t s = submit_seq; s < push_seq; ++s) {
                req & r = slots[s % N_SLOTS];
                if ((int) r.state < (int) req::state_t::ready) {
                    cv_host.wait(lk, [&r] { return (int) r.state >= (int) req::state_t::ready; });
                }
            }
        }
        submit_ready(ctx);
    }

    // full retire (fence drain points): every submitted slot's DMA is complete
    void retire_all(ggml_backend_vk_context * ctx) {
        if (!ok) {
            return;
        }
        std::lock_guard<std::mutex> lk(mtx);
        retire_seq = submit_seq;
        for (size_t i = 0; i < N_SLOTS; ++i) {
            if ((int) slots[i].state >= (int) req::state_t::submitted) {
                slots[i].state = req::state_t::free;
            }
        }
        cv_host.notify_all();
    }

private:
    // retirement of submitted slots whose DMA the timeline semaphore already
    // covers; called with mtx held
    void retire_completed(ggml_backend_vk_context * ctx, std::unique_lock<std::mutex> *) {
        if (retire_seq == submit_seq) {
            return;
        }
        uint64_t cur = 0;
        try {
            cur = ctx->device->device.getSemaphoreCounterValue(ctx->transfer_semaphore.s);
        } catch (...) {
            return;
        }
        bool retired_any = false;
        while (retire_seq < submit_seq) {
            req & r = slots[retire_seq % N_SLOTS];
            if (r.state != req::state_t::submitted || r.sem_value_at_submit > cur) {
                break;   // in-order: stop at the first slot whose DMA is not done
            }
            r.state = req::state_t::free;
            retire_seq++;
            retired_any = true;
        }
        if (retired_any) {
            cv_host.notify_all();
        }
    }

    void worker_loop() {
        // yield disk and memory bandwidth to the inference thread
        SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
        while (true) {
            size_t processed = 0;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_work.wait(lk, [&] { return stop.load(std::memory_order_relaxed) || ready_seq < push_seq; });
                if (stop.load(std::memory_order_relaxed) && ready_seq >= push_seq) {
                    return;
                }
                processed = (size_t) (ready_seq % N_SLOTS);   // in-order slot index
            }
            req & r = slots[processed];
            if (r.n > 0) {
                uint8_t * slot_base = (uint8_t *) buffer->ptr + r.staging_off;
#if defined(_WIN32)
                // page-in the source ranges with kernel help; cheap for resident pages
                {
                    WIN32_MEMORY_RANGE_ENTRY entries[MAX_ENTRIES];
                    ULONG ne = 0;
                    for (size_t i = 0; i < r.n && i < MAX_ENTRIES; ++i) {
                        entries[ne].VirtualAddress = (void *) r.data[i];
                        entries[ne].NumberOfBytes  = r.sizes[i];
                        ne++;
                    }
                    if (ne > 0) {
                        PrefetchVirtualMemory(GetCurrentProcess(), ne, entries, 0);
                    }
                }
#endif
                size_t off = 0;
                for (size_t i = 0; i < r.n; ++i) {
                    memcpy(slot_base + off, r.data[i], r.sizes[i]);
                    off += r.sizes[i];
                }
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                r.state = req::state_t::ready;
                ready_seq++;
            }
            cv_host.notify_all();
        }
    }
};
