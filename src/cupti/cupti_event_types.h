#ifndef CUPTI_EVENT_TYPES_H
#define CUPTI_EVENT_TYPES_H

#include <stdint.h>
#include <cuda_runtime_api.h>

typedef struct cupti_event_list_st {
   int stack_id;
   char *func_name;
   int cbid;
   int n_calls;
   float t_ms;
   uint64_t memcpy_bytes; 
   struct cupti_event_list_st *next;
} cupti_event_list_t;

enum {T_CUDA_COMP, T_CUDA_MEMCP};

#endif
