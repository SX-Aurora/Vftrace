#ifndef VFD_TYPES_H
#define VFD_TYPES_H

#include <stdbool.h>

typedef struct {
   int vfd_version;
   char *package_string;
   char *datestr_start;
   char *datestr_end;
   long long interval;
   unsigned int nprocesses;
   unsigned int processID;
   double runtime;
   unsigned int function_samplecount;
   unsigned int message_samplecount;
   unsigned int nstacks;
   long int samples_offset;
   long int stacks_offset;
   // TODO: hardware counters
} vfd_header_t;

typedef struct {
    char *name;
    int caller;
    int ncallees;
    int *callees;
    bool precise;
} stack_t;

#endif
