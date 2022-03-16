#ifndef PROCESS_TYPES_H
#define PROCESS_TYPES_H

#include "stacks.h"
#include "threads.h"

typedef struct {
   unsigned int nprocesses;
   unsigned int processID;
   stacktree_t stacktree;
   threadtree_t threadtree;
} process_t;

#endif
