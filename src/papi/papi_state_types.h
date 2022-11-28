#ifndef PAPI_STATE_TYPES_H
#define PAPI_STATE_TYPES_H

#include <stdbool.h>

#include <papi.h>

typedef struct {
   int eventset;
   int n_available_events;
   unsigned int *event_codes;
   char **event_names;
   char **event_units;
   char **event_descriptions;
} papi_state_t;

#endif
