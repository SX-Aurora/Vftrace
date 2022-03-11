#ifndef THREAD_TYPES_H
#define THREAD_TYPES_H

#include <stdbool.h>

#include "threadstack_types.h"

typedef struct {
   int level;
   int thread_num;
   bool master;
   threadstacklist_t stacklist;
   int parent_thread;
   int threadID;
   int maxsubthreads;
   int nsubthreads;
   int *subthreads;
} thread_t;

typedef struct {
   int nthreads;
   int maxthreads;
   thread_t *threads;
} threadtree_t;

#endif
