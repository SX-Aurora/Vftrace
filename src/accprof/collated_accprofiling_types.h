#ifndef COLLATED_ACCPROFILING_TYPES_H
#define COLLATED_ACCPROFILING_TYPES_H

#include <stdint.h>

#include "acc_prof.h"

typedef struct {
  int event_type;
  uint64_t region_id;
  int line_start;
  int line_end;
  long long copied_bytes;
  char *source_file;
  char *func_name;
  char *var_name; // Only for data events: Name of moved variable
  char *kernel_name; // Only for launch events: Name of auto-generated device kernel.
  long long overhead_nsec;
  int on_nranks;
  int ncalls[2];
  int avg_ncalls[2];
  int max_ncalls[2];
  int max_on_rank[2];
  int min_ncalls[2];
  int min_on_rank[2];
} collated_accprofile_t;

#endif
