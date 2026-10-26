#include "model_file.h"

#include <stdio.h>

#include <new>

struct qrt_model_file {
    FILE * f;
};

qrt_model_file * qrt_model_file_open(const char * path, const char * mode) {
    FILE * f = fopen(path, mode);
    if (f == NULL) {
        return NULL;
    }
    qrt_model_file * m = new (std::nothrow) qrt_model_file{f};
    if (m == NULL) {
        fclose(f);
    }
    return m;
}

int qrt_model_file_close(qrt_model_file * f) {
    if (f == NULL) {
        return EOF;
    }
    const int rc = fclose(f->f);
    delete f;
    return rc;
}

size_t qrt_model_file_write(qrt_model_file * f, const void * ptr, size_t size, size_t count) {
    return fwrite(ptr, size, count, f->f);
}

int qrt_model_file_getc(qrt_model_file * f) {
    return fgetc(f->f);
}

int qrt_model_file_error(qrt_model_file * f) {
    return ferror(f->f) ? 1 : 0;
}
