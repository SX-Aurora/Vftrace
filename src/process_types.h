#ifndef PROCESS_TYPES_H
#define PROCESS_TYPES_H

#include "stack_types.h"
#include "stacks.h"
#include "collated_stack_types.h"
#include "threads.h"

#ifdef _OMP
#include <omp.h>
#endif

typedef struct {
   int nprocesses;
   int processID;
   stacktree_t stacktree;
   collated_stacktree_t collated_stacktree;
   threadtree_t threadtree;
#ifdef _OMP
   omp_lock_t threadlock;
#endif
} process_t;

#endif
