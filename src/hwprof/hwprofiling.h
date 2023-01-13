#ifndef HWPROFILING_H
#define HWPROFILING_H

#include <stdbool.h>

#include "hwprofiling_types.h"

hwprofile_t vftr_new_hwprofiling();

long long *vftr_get_papi_counters();

void vftr_accumulate_hwprofiling (hwprofile_t *prof, long long *counters, bool invert_sign);

void vftr_update_stacks_exclusive_counters (stacktree_t *stacktree_ptr);

void vftr_update_stacks_hw_observables (stacktree_t *stacktree_ptr);

void vftr_hwprofiling_free (hwprofile_t *prof_ptr);
#endif
