#ifndef HWPROF_LIKWID_H
#define HWPROF_LIKWID_H

#include "hwprof_state_types.h"

void vftr_likwid_init (hwprof_state_t *state);

void vftr_likwid_finalize ();

long long *vftr_get_likwid_counters ();
#endif
