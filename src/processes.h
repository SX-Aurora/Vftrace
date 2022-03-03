#ifndef PROCESSES_H
#define PROCESSES_H

#include "timer.h"
#include "stacks.h"
#include "threads.h"

typedef struct {
   int nprocesses;
   int processID;
   reftime_t reftime;
   stacktree_t stacktree;
   threadtree_t threadtree;
} process_t;

process_t vftr_new_process(reftime_t reftime);

void vftr_process_free(process_t *process_ptr);

#endif
