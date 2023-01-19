#ifndef VFD_TYPES_H
#define VFD_TYPES_H

#include <stdbool.h>

typedef struct {
   int vfd_version;
   char *package_string;
   char *datestr_start;
   char *datestr_end;
   long long interval;
   int nprocesses;
   int processID;
   int nthreads;
   double runtime;
   unsigned int function_samplecount;
   unsigned int message_samplecount;
   unsigned int nstacks;
   unsigned int n_hw_counters;
   unsigned int n_hw_observables;
   long int samples_offset;
   long int stacks_offset;
   long int threadtree_offset;
   long int hwprof_offset;
   // TODO: hardware counters
} vfd_header_t;

typedef struct {
    char *name;
    int caller;
    int ncallees;
    int *callees;
    bool precise;
} vftr_stack_t;

typedef struct {
   int threadID;
   int parent_thread;
   int nchildren;
   int *children;
   int level;
} thread_t;

#endif
