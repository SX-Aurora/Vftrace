#ifndef TIMER_H
#define TIMER_H

#ifdef _DEBUG
#include <stdio.h>
#endif

// get the current time in micro seconds since
// the reference time point
long long vftr_get_runtime_usec();

// get the number of elapsed clock counts
unsigned long long vftr_get_cycles();

// set the local reference time to which all 
// timedifferences are measured
void vftr_set_local_ref_time();

#ifdef _DEBUG
void vftr_print_date_str(FILE *fp);
#endif

#endif