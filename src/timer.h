#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>

#include "timer_types.h"

// get the current time in micro seconds since
// the reference time point
long long vftr_get_runtime_nsec();

// get the number of elapsed clock counts
unsigned long long vftr_get_cycles();

// set the local reference time to which all
// timedifferences are measured
void vftr_set_local_ref_time();

char *vftr_get_date_str();

void vftr_timestrings_free(time_strings_t *timestrings);

void vftr_print_date_str(FILE *fp);

#endif
