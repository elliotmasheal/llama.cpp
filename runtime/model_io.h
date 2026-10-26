#pragma once

#include <stddef.h>
#include <stdint.h>

// The model run's file format edge: reading token ids, writing the QRTLOG1 logits dump and
// writing generated ids. Keeping stdio here means the run and the checks carry no file
// handles, and the format rules have exactly one home.

// One QRTLOG1 v1 dump: 24 B header, the tokens of the covered range plus one, then
// n_rows f32 rows (bench/tools/quality/README.md section 3).
int qrt_model_write_dump(const char * path, uint32_t V, uint32_t n_rows, uint32_t start,
                         const uint32_t * ids, const float * logits, size_t row_stride,
                         char * err, size_t err_size);

// Generated ids, one decimal per line.
int qrt_model_write_ids(const char * path, const uint32_t * ids, size_t count,
                        char * err, size_t err_size);
