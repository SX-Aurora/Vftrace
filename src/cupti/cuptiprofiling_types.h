#ifndef CUPTIPROFILING_TYPES_H
#define CUPTIPROFILING_TYPES_H

#include <stdint.h>
#include <cuda_runtime_api.h>

typedef struct {
   cudaEvent_t start;
   cudaEvent_t stop;
   int cbid;
   int n_calls;
   float t_ms;
   long long t_vftr;
   size_t memcpy_bytes[2];
} cuptiprofile_t;
#endif
