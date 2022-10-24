#ifndef COLLATE_ACCPROFILES_H
#define COLLATE_ACCPROFILES_h

#include "collated_stack_types.h"

void vftr_collate_accprofiles (collated_stacktree_t *collstacktree_ptr,
                               stacktree_t *stacktree_ptr,
                               int myrank, int nranks, int *nremote_profiles);

#endif
