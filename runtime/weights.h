#pragma once

#include <stddef.h>
#include <stdint.h>

// Host side of the repacked expert store (design 5.1 `weights`, M2-3).
// Reads <prefix>.qwen36rt.idx into a page table and hands out expert byte
// ranges from <prefix>.qwen36rt.bin. No VRAM is touched here.

struct qrt_weights;

struct qrt_expert_loc {
    uint64_t offset;   // relative to the payload start
    uint64_t bytes;
    uint64_t gate;
    uint64_t up;
    uint64_t down;
};

// prefix is the path without .qwen36rt.{bin,idx}.
int qrt_weights_open(const char * prefix, qrt_weights ** out, char * err, size_t err_size);
void qrt_weights_close(qrt_weights * w);

uint32_t qrt_weights_layers(const qrt_weights * w);
uint32_t qrt_weights_experts(const qrt_weights * w);
uint64_t qrt_weights_payload_bytes(const qrt_weights * w);
uint32_t qrt_weights_records(const qrt_weights * w);
const char * qrt_weights_arch(const qrt_weights * w);

// Returns -1 with a message when the pair is unknown or out of range.
int qrt_weights_locate(const qrt_weights * w, uint32_t layer, uint32_t expert,
                       qrt_expert_loc * out, char * err, size_t err_size);

// Reads one expert block (gate||up||down) into dst; cap must hold loc.bytes.
int qrt_weights_read_expert(const qrt_weights * w, uint32_t layer, uint32_t expert,
                            void * dst, size_t cap, char * err, size_t err_size);
