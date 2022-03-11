#ifndef THREADSTACK_TYPES_H
#define THREADSTACK_TYPES_H

#include "profiling_types.h"

typedef struct {
   int stackID;
   int recursion_depth;
   profile_t profiling;
} threadstack_t;

typedef struct {
   int nstacks;
   int maxstacks;
   threadstack_t *stacks;
} threadstacklist_t;

#endif
