#pragma once

#include "vk_dev.h"

// Host <-> device transfer: the staging ring and its copies. A module that only moves bytes
// includes this and not the pass or pipeline APIs.

// Host-visible staging buffer plus its own command buffer and fence. Mapped once
// at creation; a copy is one vkCmdCopyBuffer and one submit+wait. M2-5 keeps two
// of them to overlap the upload of block n+1 with the compute of block n.
struct qrt_vk_staging;
int qrt_vk_staging_create(qrt_vk_dev * dev, uint64_t bytes, qrt_vk_staging ** out,
                          char * err, size_t err_size);
void qrt_vk_staging_destroy(qrt_vk_dev * dev, qrt_vk_staging * stg);
uint64_t qrt_vk_staging_bytes(const qrt_vk_staging * stg);
// One line saying which memory type and heap the staging buffer landed in.
const char * qrt_vk_staging_label(const qrt_vk_staging * stg);

// Copies host memory into a device buffer, or back out of it, and returns only
// after the fence signals. Both report failure instead of aborting (design 9).
int qrt_vk_upload(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                  uint64_t dst_offset, const void * src, uint64_t bytes,
                  char * err, size_t err_size);
int qrt_vk_download(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                    uint64_t src_offset, void * dst, uint64_t bytes,
                    char * err, size_t err_size);

// The variants the engine uses: the caller writes straight into the mapping (a
// file read lands there, so one memcpy disappears) and only the copy runs here.
void * qrt_vk_staging_map(qrt_vk_staging * stg);
int qrt_vk_upload_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size);
int qrt_vk_download_mapped(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer src,
                           uint64_t src_offset, uint64_t bytes, char * err, size_t err_size);

// Several device copies from one staging buffer in one submit (M3-2a): the sources
// are packed back to back into the mapping by the caller, and the regions may target
// different buffers and offsets (the decode cache slots are not contiguous). The
// sum of bytes must fit the staging buffer. Waits for the submit before returning.
struct qrt_vk_copy_region {
    const void * src;          // inside the staging mapping
    VkBuffer     dst;
    uint64_t     dst_offset;
    uint64_t     bytes;
};
int qrt_vk_upload_regions(qrt_vk_dev * dev, qrt_vk_staging * stg,
                          const qrt_vk_copy_region * regions, uint32_t count,
                          char * err, size_t err_size);

// Fire and forget: the copy is in flight when this returns. Keep one staging
// buffer per copy in flight and wait before touching its mapping again; this is
// how M2-5 overlaps the upload of block n+1 with the compute of block n.
int qrt_vk_upload_submit(qrt_vk_dev * dev, qrt_vk_staging * stg, VkBuffer dst,
                         uint64_t dst_offset, uint64_t bytes, char * err, size_t err_size);
int qrt_vk_staging_wait(qrt_vk_dev * dev, qrt_vk_staging * stg, char * err, size_t err_size);
