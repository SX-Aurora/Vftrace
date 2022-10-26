#ifndef ACCPROFILING_TYPES_H
#define ACCPROFILING_TYPES_H

#include "acc_prof.h"

typedef struct {
   acc_event_t event_type; 
   int line_start;
   int line_end;
   size_t copied_bytes; // Only for data events: Amount of data moved
   char *source_file;
   char *var_name; // Only for data events: Name of moved variable
   char *kernel_name; // Only for launch events: Name of auto-generated device kernel.
} accprofile_t;

#endif
