#ifndef OMPPROFILING_H
#define OMPPROFILING_H

#include <stdio.h>

#include "collated_stack_types.h"
#include "ompprofiling_types.h"
#include "environment_types.h"

ompprofile_t vftr_new_ompprofiling();

void vftr_ompprofiling_free(ompprofile_t *prof_ptr);

#ifdef _OMP

void vftr_accumulate_ompprofiling_overhead(ompprofile_t *prof,
                                           long long overhead_nsec);

void vftr_print_ompprofiling(FILE *fp, ompprofile_t ompprof);

#endif

long long *vftr_get_total_omp_overhead(stacktree_t stacktree, int nthreads);

long long vftr_get_total_collated_omp_overhead(collated_stacktree_t stacktree);

#endif
