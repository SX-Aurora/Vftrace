#ifndef THREADS_H
#define THREADS_H

#include <stdbool.h>

#include "stacks.h"

typedef struct thread_type thread_t;
struct thread_type {
   int level;
   int thread_num;
   stack_t *current_stack;
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

#endif
