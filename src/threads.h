#ifndef THREADS_H
#define THREADS_H

#include <stdio.h>
#include <stdbool.h>

#include "thread_types.h"

threadtree_t vftr_new_threadtree();

int vftr_new_thread(int parent_thread_id,
                    threadtree_t *threadtree_ptr);

void vftr_threadtree_free(threadtree_t *threadtree_ptr);

int vftr_get_thread_level();

int vftr_get_thread_num();

int vftr_get_ancestor_thread_num(int level);

thread_t *vftr_get_my_thread(threadtree_t *threadtree_ptr);

void vftr_print_threadtree(FILE *fp, threadtree_t threadtree);

void vftr_print_threadlist(FILE *fp, threadtree_t threadtree);

#endif
