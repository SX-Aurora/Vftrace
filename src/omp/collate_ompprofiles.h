#ifndef COLLATE_OMPPROFILES_H
#define COLLATE_OMPPROFILES_H

#include <stdlib.h>

#include "stack_types.h"
#include "collated_stack_types.h"

void vftr_collate_ompprofiles(collated_stacktree_t *collstacktree_ptr,
                              stacktree_t *stacktree_ptr,
                              int myrank, int nranks,
                              int *nremote_profiles);

#endif
