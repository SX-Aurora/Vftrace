#ifndef CUPTI_EVENTS_H
#define CUPTI_EVENTS_H

#include <stdbool.h>

#include "cupti_event_types.h"

cupti_event_list_t *new_cupti_event (char *func_name, int cbid, float t_ms, uint64_t memcpy_bytes);

void acc_cupti_event (cupti_event_list_t *event, float t_ms, uint64_t memcpy_bytes);

bool cupti_event_is_compute (cupti_event_list_t *event);

bool cupti_event_is_memcpy (cupti_event_list_t *event);

bool cupti_event_is_other (cupti_event_list_t *event);
#endif
