#ifndef STACK_TYPES_H
#define STACK_TYPES_H

#include <time.h>

typedef struct {
   char *name;
   int id;
   // calling function index
   int caller;
   // called function indices
   int maxcallees;
   int ncallees;
   int *callees;
   // profiling
   int ncalls;
   long long time_nsec;
   long long time_excl_nsec;
   struct timespec t_enter;
   struct timespec t_leave;
} vftr_stack_t;

typedef struct {
   int nstacks;
   int maxstacks;
   vftr_stack_t *stacks;
} stacktree_t;

#endif
