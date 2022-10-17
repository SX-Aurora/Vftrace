#ifndef COLLATE_STACKS_H
#define COLLATE_STACKS_H

#include "stack_types.h"
#include "collated_stack_types.h"

collated_stacktree_t vftr_new_empty_collated_stacktree();

collated_stacktree_t vftr_collate_stacks(stacktree_t *stacktree_ptr);

void vftr_collated_stacktree_realloc(collated_stacktree_t *stacktree_ptr);

void vftr_collated_stacktree_free(collated_stacktree_t *stacktree_ptr);

collated_stacktree_t vftr_collated_stacktree_group_by_name(
   collated_stacktree_t *stacktree_ptr);

void vftr_print_collated_stacklist(FILE *fp, collated_stacktree_t stacktree);

char *vftr_get_collated_stack_string(collated_stacktree_t stacktree,
                                     int stackid, bool show_precise);

#endif
