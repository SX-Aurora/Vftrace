#ifndef CUPTI_STATE_TYPES_H
#define CUPTI_STATE_TYPES_H

#include <cupti_callbacks.h>
#include "cupti_event_types.h"

typedef struct {
   int n_devices;
   //CUpti_SubscriberHandle subscriber;
   cupti_event_list_t *event_buffer;
} cupti_state_t;
#endif
