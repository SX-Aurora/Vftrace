#ifndef CUPTI_EVENT_LIST_H
#define CUPTI_EVENT_LIST_H

#include <stdbool.h>

#include "cupti_event_types.h"

cupti_event_list_t *new_cupti_event (char *func_name, int cbid, float t_ms,
                                     int mem_dir, size_t memcpy_bytes);

void acc_cupti_event (cupti_event_list_t *event, float t_ms, int mem_dir, size_t memcpy_bytes);

bool cupti_event_belongs_to_class (cupti_event_list_t *event, int cbid_class);
#endif
