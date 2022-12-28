#ifndef COLLATED_CUDAPROFILING_TYPES_H
#define COLLATED_CUDAPROFILING_TYPES_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
   int cbid;
   int n_calls;
   float t_ms;
   size_t memcpy_bytes[2];
   long long overhead_nsec;
} collated_cudaprofile_t;
#endif
