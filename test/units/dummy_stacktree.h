#ifndef DUMMY_STACKTREE_H
#define DUMMY_STACKTREE_H

#include "stack_types.h"

void vftr_init_dummy_stacktree (uint64_t t_call, uint64_t t_overhead);
void vftr_register_dummy_stack (char *stackstring, int thread_id, uint64_t t_call, uint64_t t_overhead);
stacktree_t vftr_get_dummy_stacktree();
#endif
