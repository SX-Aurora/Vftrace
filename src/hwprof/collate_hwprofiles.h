#ifndef COLLATE_HWPROFILES_H
#define COLLATE_HWPROFILES_H

void vftr_collate_hwprofiles (collated_stacktree_t *collstacktree_ptr,
			      stacktree_t *stacktree_ptr,
			      int myrank, int nranks, int *nremote_profiles);

#endif
