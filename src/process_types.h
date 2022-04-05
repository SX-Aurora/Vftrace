#ifndef PROCESS_TYPES_H
#define PROCESS_TYPES_H

#include "stack_types.h"
#include "stacks.h"
#include "collated_stack_types.h"
#include "threads.h"

typedef struct {
   unsigned int nprocesses;
   unsigned int processID;
   stacktree_t stacktree;
   collated_stacktree_t collated_stacktree;
   threadtree_t threadtree;
} process_t;

#endif
