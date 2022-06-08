#ifndef CALLPROFILING_H
#define CALLPROFILING_H

#include <stdio.h>

#include "callprofiling_types.h"
#include "stack_types.h"

#include "callprofiling.h"

callProfile_t vftr_new_callprofiling();

void vftr_accumulate_callprofiling(callProfile_t *prof,
                                   int calls,
                                   long long cycles,
                                   long long time_usec);

void vftr_callprofiling_free(callProfile_t *callprof_ptr);

void vftr_update_stacks_exclusive_time(int nstacks, stack_t *stacks);

void vftr_print_callprofiling(FILE *fp, callProfile_t callprof);

#endif
