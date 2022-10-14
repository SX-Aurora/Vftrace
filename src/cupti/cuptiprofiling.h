#ifndef CUPTIPROFILING_H
#define CUPTIPROFILING_H

#include <cupti.h>
#include "cuptiprofiling_types.h"

cuptiprofile_t vftr_new_cuptiprofiling();
void vftr_cuptiprofiling_free (cuptiprofile_t *prof_ptr);

void vftr_accumulate_cuptiprofiling (cuptiprofile_t *prof, int cbid, int n_calls,
                                     float t_ms, int mem_dir, uint64_t memcpy_bytes);

cuptiprofile_t vftr_add_cuptiprofiles(cuptiprofile_t profA, cuptiprofile_t profB);

#endif
