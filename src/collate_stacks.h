#ifndef COLLATE_STACKS_H
#define COLLATE_STACKS_H

#include "stack_types.h"
#include "collated_stack_types.h"

collated_stacktree_t vftr_new_collated_stacktree();

collated_stacktree_t vftr_collate_stacks(stacktree_t *stacktree_ptr);

void vftr_collated_stacktree_free(collated_stacktree_t *stacktree_ptr);

void vftr_print_collated_stacklist(FILE *fp, collated_stacktree_t stacktree);

#endif
