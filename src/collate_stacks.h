#ifndef COLLATE_STACKS_H
#define COLLATE_STACKS_H

#include "stack_types.h"

void vftr_collate_stacks(stacktree_t *stacktree_ptr);

void vftr_collated_stacktree_free(int *nstacks_ptr,
                                  collated_stack_t **stacks_ptr);

#endif
