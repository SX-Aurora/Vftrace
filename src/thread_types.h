#ifndef THREAD_TYPES_H
#define THREAD_TYPES_H

#include <stdbool.h>

#include "threadstacks.h"

typedef struct {
   int level;
   int thread_num;
   bool master;
   threadstacklist_t stacklist;
   int parent_thread;
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
