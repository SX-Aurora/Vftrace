#ifndef CUPTI_EVENT_TYPES_H
#define CUPTI_EVENT_TYPES_H

#include <stdint.h>
#include <cuda_runtime_api.h>

typedef struct cupti_event_list_st {
   char *func_name;
   int cbid;
   int n_calls;
   float t_ms;
   size_t memcpy_bytes[2];
   struct cupti_event_list_st *next;
} cupti_event_list_t;

enum {T_CUPTI_COMP, T_CUPTI_MEMCP, T_CUPTI_OTHER};
enum {CUPTI_NOCOPY=-1,CUPTI_COPY_IN=0, CUPTI_COPY_OUT=1};

#endif
