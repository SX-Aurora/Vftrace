#ifndef THREADS_H
#define THREADS_H

#include <stdbool.h>

typedef struct thread_type thread_t;
struct thread_type {
   int level;
   int thread_num;
   int current_stackID;
   bool master;
   thread_t *parent_thread;
   int maxsubthreads;
   int nsubthreads;
   thread_t **subthreads;
};

typedef struct {
   int nthreads;
   int maxthreads;
   thread_t *threads;
} threadtree_t;

threadtree_t vftr_new_threadtree(stack_t *rootstack);

void vftr_threadtree_free(threadtree_t *threadtree_ptr);

thread_t *vftr_get_my_thread(threadtree_t threadtree);

#endif
