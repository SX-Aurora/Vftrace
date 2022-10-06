#ifndef CUPTIPROFILING_TYPES_H
#define CUPTIPROFILING_TYPES_H

#include <stdint.h>
#include <cuda_runtime_api.h>

#include "cupti_event_types.h"

typedef struct {
   cupti_event_list_t *events;
   cudaEvent_t start;
   cudaEvent_t stop;
} cuptiprofile_t;
#endif
