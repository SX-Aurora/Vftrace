#ifndef CUPTI_STATE_TYPES_H
#define CUPTI_STATE_TYPES_H

#include <cupti_callbacks.h>
#include "cupti_event_types.h"

typedef struct {
   int n_devices;
   cupti_event_list_t *event_buffer;
} cupti_state_t;
#endif
