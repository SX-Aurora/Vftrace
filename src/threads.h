#ifndef THREADS_H
#define THREADS_H

#include <stdbool.h>

#include "thread_types.h"

threadtree_t vftr_new_threadtree();

void vftr_threadtree_free(threadtree_t *threadtree_ptr);

thread_t *vftr_get_my_thread(threadtree_t threadtree);

#endif
