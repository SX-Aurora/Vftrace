#ifndef HWPROF_STATE_TYPES_H
#define HWPROF_STATE_TYPES_H

#include <stdbool.h>

#include "hwprof_ve.h"
#include "calculator.h"

enum {HWC_NONE, HWC_DUMMY, HWC_PAPI, HWC_VE};

typedef struct {
   char *name;
} vftr_counter_t;

typedef struct {
   int eventset;
} papi_state_t;

typedef struct {
  //const char **ve_hwc_names;
  int *active_counters; 
} veprof_state_t;

typedef struct {
   int hwc_type;
   int n_counters;
   int n_observables;
   vftr_counter_t *counters;
   papi_state_t papi;
   veprof_state_t veprof;
   vftr_calculator_t calculator;
} hwprof_state_t;

#endif
