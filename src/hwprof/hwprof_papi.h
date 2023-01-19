#ifndef HWPROF_PAPI_H
#define HWPROF_PAPI_H

#include <stdio.h>
#include <stdlib.h>

#include "hwprof_state_types.h"

void vftr_show_papi_components (FILE *fp);

void vftr_papi_init(hwprof_state_t *state);

void vftr_papi_finalize();

long long *vftr_get_papi_counters ();

#endif
