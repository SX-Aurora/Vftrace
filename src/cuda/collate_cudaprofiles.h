#ifndef COLLATE_CUDAPROFILES_H
#define COLLATE_CUDAPROFILES_H

#include "collated_stack_types.h"

void vftr_collate_cudaprofiles (collated_stacktree_t *collstacktree_ptr, 
  				stacktree_t *stacktree_ptr,
 				int myrank, int nranks, int *nremote_profiles);

#endif
