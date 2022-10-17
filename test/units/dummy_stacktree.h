#ifndef DUMMY_STACKTREE_H
#define DUMMY_STACKTREE_H

#include "stack_types.h"

stacktree_t vftr_init_dummy_stacktree(uint64_t t_call, uint64_t t_overhead);

void vftr_register_dummy_stack(stacktree_t *stacktree_ptr,
                               char *stackstring,
                               int thread_id,
                               uint64_t t_call,
                               uint64_t t_overhead);

#endif
