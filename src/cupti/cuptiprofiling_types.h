#ifndef CUPTIPROFILING_TYPES_H
#define CUPTIPROFILING_TYPES_H

#include <stdint.h>

typedef struct {
   //char *cupti_object_name;
   int n_calls;
   float t_compute;
   float t_memcpy;
   uint64_t copied_bytes;  
} cuptiprofile_t;
#endif
