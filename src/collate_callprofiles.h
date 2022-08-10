#ifndef COLLATE_CALLPROFILES_H
#define COLLATE_CALLPROFILES_H

#include <stdlib.h>

#include "stack_types.h"
#include "collated_stack_types.h"

void vftr_collate_callprofiles(collated_stacktree_t *collstacktree_ptr,
                               stacktree_t *stacktree_ptr,
                               int myrank, int nranks,
                               int *nremote_profiles);

#endif
