#ifndef CUDAPROFILING_TYPES_H
#define CUDAPROFILING_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime_api.h>

typedef struct {
   cudaEvent_t start;
   cudaEvent_t stop;
   int cbid;
   int n_calls[2];
   float t_ms;
   long long memcpy_bytes[2];
   long long overhead_nsec;
} cudaprofile_t;
#endif
