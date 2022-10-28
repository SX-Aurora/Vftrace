#ifndef VFTRACE_STATE_H
#define VFTRACE_STATE_H

#include <stdbool.h>

#include "hook_types.h"
#include "configuration_types.h"
#include "symbol_types.h"
#include "process_types.h"
#include "sampling_types.h"
#include "timer_types.h"
#include "size_types.h"
#ifdef _OMP
#include "omp/omp_state_types.h"
#endif
#ifdef _CUPTI
#include "cupti/cupti_state_types.h"
#endif
#ifdef _MPI
#include "mpi/utils/mpi_state_types.h"
#endif

// tracing state of vftrace
typedef enum {
   uninitialized,
   off,
   on,
   paused
} state_t;

// main datatype to store everything
typedef struct {
   hooks_t hooks; // collection of function pointers
                  // where vftrace intercepts the program flow
   config_t config; // set of all options
   symboltable_t symboltable; // list of function symbols
   process_t process; // all of the dynamic process data
   state_t state; // current state of vftrace
   sampling_t sampling; // Filehandle and data required to handle vfd-file sampling
   time_strings_t timestrings; // start and end time in string form
#ifdef _OMP
   omp_state_t omp_state;
#endif
#ifdef _CUPTI
   cupti_state_t cupti_state;
#endif
#ifdef _MPI
   mpi_state_t mpi_state;
#endif
   vftr_size_t size;
} vftrace_t;

extern vftrace_t vftrace;

unsigned long long vftr_sizeof_vftrace_t(vftrace_t vftrace_state);

#endif
