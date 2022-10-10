#ifndef COLLATE_CUPTIPROFILES_H
#define COLLATE_CUPTIPROFILES_H

#include "collated_stack_types.h"

void vftr_collate_cuptiprofiles (collated_stacktree_t *collstacktree_ptr, 
  				 stacktree_t *stacktree_ptr,
 				 int myrank, int nranks, int *nremote_profiles);

#endif
