#ifndef PROFILING_H
#define PROFILING_H

#include <stdbool.h>

#include "profiling_types.h"
#include "stack_types.h"

callProfile_t vftr_new_callprofiling();

profile_t vftr_new_profiling();

void vftr_accumulate_profiling(bool master, profile_t *stackprof,
                               profile_t *threadprof);

void vftr_callprofiling_free(callProfile_t *callprof_ptr);

void vftr_update_stacks_exclusive_time(int nstacks, stack_t *stacks);

void vftr_profiling_free(profile_t *prof_ptr);

#endif
