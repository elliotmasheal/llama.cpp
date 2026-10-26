#pragma once

#include "vk_dev.h"

// One layer pass (design 4.6, DECISIONS 64.2/68): every module of a layer records its
// dispatches into the same command buffer, so the layer costs one submit instead of one
// per module. The modules keep their own descriptor pools; a module attaches to a ring,
// records, and the caller submits and waits once. Two rings so the next pass can be
// recorded while the current one is in flight.

struct qrt_vk_cmd;
int qrt_vk_cmd_create(qrt_vk_dev * dev, uint32_t rings, qrt_vk_cmd ** out,
                      char * err, size_t err_size);
void qrt_vk_cmd_destroy(qrt_vk_dev * dev, qrt_vk_cmd * c);
uint32_t qrt_vk_cmd_rings(const qrt_vk_cmd * c);
// Waits a previous submission on this ring, resets its fence and begins the buffer.
int qrt_vk_cmd_begin(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
VkCommandBuffer qrt_vk_cmd_buffer(qrt_vk_cmd * c, uint32_t ring);
// Compute-to-compute memory barrier: the next dispatch sees the previous ones' writes.
int qrt_vk_cmd_barrier(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
// TRANSFER -> COMPUTE barrier, for a vkCmdFillBuffer the same pass recorded.
int qrt_vk_cmd_barrier_transfer(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
void qrt_vk_cmd_record_barrier_transfer(VkCommandBuffer cmd);
// Records a compute->compute barrier into a buffer another module is recording into.
void qrt_vk_cmd_record_barrier(VkCommandBuffer cmd);
// How many dispatches the pass recorded, for the readings.
void qrt_vk_cmd_count(qrt_vk_cmd * c, uint32_t dispatches);
uint32_t qrt_vk_cmd_dispatches(const qrt_vk_cmd * c);
int qrt_vk_cmd_submit(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
int qrt_vk_cmd_wait(qrt_vk_cmd * c, uint32_t ring, char * err, size_t err_size);
