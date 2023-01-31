#ifndef VFTRACE_STATE_H
#define VFTRACE_STATE_H

#include <stdbool.h>
#include <signal.h>

#include "hook_types.h"
#include "configuration_types.h"
#include "symbol_types.h"
#include "process_types.h"
#include "sampling_types.h"
#include "timer_types.h"
#include "size_types.h"
#include "hwprof_state_types.h"
#ifdef _OMP
#include "omp_state_types.h"
#endif
#ifdef _CUDA
#include "cuda_state_types.h"
#endif
#ifdef _ACCPROF
#include "accprof_state_types.h"
#endif
#ifdef _MPI
#include "mpi_state_types.h"
#include "minmax_summary_types.h"
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
   struct sigaction signals[NSIG];
   int signal_received;
   hwprof_state_t hwprof_state;
#ifdef _OMP
   omp_state_t omp_state;
#endif
#ifdef _CUDA
   cuda_state_t cuda_state;
#endif
#ifdef _ACCPROF
   accprof_state_t accprof_state;
#endif
#ifdef _MPI
   mpi_state_t mpi_state;
#endif
   vftr_size_t size;
} vftrace_t;

extern vftrace_t vftrace;

unsigned long long vftr_sizeof_vftrace_t(vftrace_t vftrace_state);

#endif
