#ifndef CALLPROFILING_H
#define CALLPROFILING_H

#include <stdio.h>

#include "callprofiling_types.h"
#include "stack_types.h"

#include "callprofiling.h"

callprofile_t vftr_new_callprofiling();

void vftr_accumulate_callprofiling(callprofile_t *prof,
                                   int calls,
                                   long long time_nsec);

void vftr_accumulate_callprofiling_overhead(callprofile_t *prof,
                                            long long overhead_nsec);

void vftr_update_stacks_exclusive_time(stacktree_t *stacktree_ptr);

long long *vftr_get_total_call_overhead(stacktree_t stacktree, int nthreads);

void vftr_callprofiling_free(callprofile_t *callprof_ptr);

void vftr_print_callprofiling(FILE *fp, callprofile_t callprof);

#endif
