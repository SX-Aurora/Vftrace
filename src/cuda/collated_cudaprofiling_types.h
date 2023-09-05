#ifndef COLLATED_CUDAPROFILING_TYPES_H
#define COLLATED_CUDAPROFILING_TYPES_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
   int cbid;
   int n_calls[2];
   float t_ms;
   long long memcpy_bytes[2];
   long long overhead_nsec;
   int on_nranks;
   int avg_ncalls[2];
   int max_ncalls[2];
   int max_on_rank[2];
   int min_ncalls[2];
   int min_on_rank[2];
} collated_cudaprofile_t;
#endif
