#ifndef COLLATED_CALLPROFILING_H
#define COLLATED_CALLPROFILING_H

#include <stdlib.h>
#include <stdio.h>

#include "collated_callprofiling_types.h"
#include "collated_stack_types.h"

collated_callProfile_t vftr_new_collated_callprofiling();

long long vftr_get_total_collated_call_overhead(collated_stacktree_t stacktree);

void vftr_collated_callprofiling_free(collated_callProfile_t *callprof_ptr);

void vftr_print_collated_callprofiling(FILE *fp, collated_callProfile_t callprof);

#endif