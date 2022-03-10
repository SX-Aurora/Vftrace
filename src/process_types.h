#ifndef PROCESS_TYPES_H
#define PROCESS_TYPES_H

#include "stacks.h"
#include "threads.h"

typedef struct {
   int nprocesses;
   int processID;
   stacktree_t stacktree;
   threadtree_t threadtree;
} process_t;

#endif
