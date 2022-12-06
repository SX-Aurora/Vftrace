#ifndef PAPIPROFILING_H
#define PAPIPROFILING_H

#include <stdbool.h>

#include "papiprofiling_types.h"

papiprofile_t vftr_new_papiprofiling();

long long *vftr_get_papi_counters();

void vftr_accumulate_papiprofiling (papiprofile_t *prof, long long *counters, bool invert_sign);

void vftr_papiprofiling_free (papiprofile_t *prof_ptr);
#endif
