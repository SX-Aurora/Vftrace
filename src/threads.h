#ifndef THREADS_H
#define THREADS_H

#include <stdbool.h>

typedef struct {
   int level;
   int thread_num;
   int current_stackID;
   bool master;
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

threadtree_t vftr_new_threadtree(stack_t *rootstack);

void vftr_threadtree_free(threadtree_t *threadtree_ptr);

thread_t *vftr_get_my_thread(threadtree_t threadtree);

#endif
