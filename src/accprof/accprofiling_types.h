#ifndef ACCPROFILING_TYPES_H
#define ACCPROFILING_TYPES_H

#include <stdint.h>

#include "acc_prof.h"

typedef struct {
   int event_type; 
   uint64_t region_id;
   int line_start;
   int line_end;
   long long copied_bytes; // Only for data events: Amount of data moved
   char *source_file;
   char *func_name;
   char *var_name; // Only for data events: Name of moved variable
   char *kernel_name; // Only for launch events: Name of auto-generated device kernel.
   long long overhead_nsec;
} accprofile_t;

#endif
