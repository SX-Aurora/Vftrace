#ifndef VEDAPROFILING_H
#define VEDAPROFILING_H

#include <stdlib.h>
#include <stdio.h>

#include "vedaprofiling_types.h"
#include "vftrace_state.h"

vedaprofile_t vftr_new_vedaprofiling();

vedaprofile_t vftr_add_vedaprofiles(vedaprofile_t profA, vedaprofile_t profB);

void vftr_accumulate_veda_profiling_overhead(vedaprofile_t *prof,
                                             long long overhead_nsec);

void vftr_vedaprofiling_free(vedaprofile_t *prof_ptr);

long long *vftr_get_total_veda_overhead(stacktree_t stacktree, int nthreads);

long long vftr_get_total_collated_veda_overhead(collated_stacktree_t stacktree);

void vftr_print_vedaprofiling(FILE *fp, vedaprofile_t vedaprof);

#endif
