#ifndef HWPROF_STATE_TYPES_H
#define HWPROF_STATE_TYPES_H

#include <stdbool.h>

#ifdef _LIKWID_AVAIL
#include "likwid.h"
#endif

#include "hwprof_ve.h"
#include "calculator.h"

#define NSYM_BUILTIN 2

enum {HWC_NONE, HWC_DUMMY, HWC_PAPI, HWC_LIKWID, HWC_VE};

extern char *vftr_builtin_obs_symbols[NSYM_BUILTIN];

typedef struct {
   char *name;
   char *symbol;
} vftr_counter_t;

typedef struct {
   char *name;
   char *formula;
   char *unit;
} vftr_observable_t;

typedef struct {
   int eventset;
} papi_state_t;

typedef struct {
  double total_energy;
#ifdef _LIKWID_AVAIL
  PowerData_t pd;
#endif
} likwid_state_t;

typedef struct {
  int *active_counters; 
} veprof_state_t;

typedef struct {
   int hwc_type;
   bool active;
   int n_counters;
   int n_observables;
   vftr_counter_t *counters;
   vftr_observable_t *observables;
   papi_state_t papi;
   likwid_state_t likwid;
   veprof_state_t veprof;
   vftr_calculator_t calculator;
} hwprof_state_t;

#endif
