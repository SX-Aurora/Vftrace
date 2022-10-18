#ifndef CUPTIPROFILING_H
#define CUPTIPROFILING_H

#include <cupti.h>

#include "stack_types.h"
#include "collated_stack_types.h"

#include "cuptiprofiling_types.h"

cuptiprofile_t vftr_new_cuptiprofiling();
void vftr_cuptiprofiling_free (cuptiprofile_t *prof_ptr);

void vftr_accumulate_cuptiprofiling (cuptiprofile_t *prof, int cbid, int n_calls,
                                     float t_ms, int mem_dir, uint64_t memcpy_bytes);

void vftr_accumulate_cuptiprofiling_overhead (cuptiprofile_t *prof, long long t_nsec);

long long vftr_get_total_cupti_overhead (stacktree_t stacktree);
long long vftr_get_total_collated_cupti_overhead (collated_stacktree_t stacktree);

cuptiprofile_t vftr_add_cuptiprofiles(cuptiprofile_t profA, cuptiprofile_t profB);

#endif
