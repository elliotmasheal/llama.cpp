#pragma once

#include <stddef.h>

// The model module's file port. model_io.cpp talks to this instead of stdio, so the QRTLOG1
// dump and the token id paths can be driven against an in-memory store without replacing
// the C library symbols of the translation unit under test.

struct qrt_model_file;

// mode is "rb" or "wb". NULL on failure.
qrt_model_file * qrt_model_file_open(const char * path, const char * mode);
// Closes the handle; EOF on failure. The handle is not usable after this call.
int qrt_model_file_close(qrt_model_file * f);
// The same short-count contract as fwrite.
size_t qrt_model_file_write(qrt_model_file * f, const void * ptr, size_t size, size_t count);
int qrt_model_file_getc(qrt_model_file * f);
int qrt_model_file_error(qrt_model_file * f);
