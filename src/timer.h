#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>
#include <stdbool.h>

#include "time.h"

typedef struct {
   bool valid;
   struct timespec timestamp;
   unsigned long long cyclecount;
} reftime_t;

// get the current time in micro seconds since
// the reference time point
long long vftr_get_runtime_usec(struct timespec reftimestamp);

// get the number of elapsed clock counts
unsigned long long vftr_get_cycles();

// set the local reference time to which all 
// timedifferences are measured
reftime_t vftr_set_local_ref_time();

#ifdef _DEBUG
void vftr_print_date_str(FILE *fp);
#endif

#endif
