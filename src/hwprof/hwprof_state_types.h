#ifndef HWPROF_STATE_TYPES_H
#define HWPROF_STATE_TYPES_H

#include <stdbool.h>

#include <papi.h>

#include "calculator.h"

typedef struct {
   char *name;
} vftr_counter_t;

typedef struct {
   int n_counters;
   int n_observables;
   vftr_counter_t *counters;
   int eventset;
   vftr_calculator_t calculator;
} hwprof_state_t;

#endif
