#ifndef PAPI_STATE_TYPES_H
#define PAPI_STATE_TYPES_H

#include <stdbool.h>

#include <papi.h>

#include "papi_calculator.h"

typedef struct {
   char *name;
   bool is_native;
} vftr_counter_t;

typedef struct {
   int n_counters;
   vftr_counter_t *counters;
   int eventset;
   papi_calculator_t calculator;
} papi_state_t;

#endif
