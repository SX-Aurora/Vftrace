#ifndef COLLATE_PAPIPROFILES_H
#define COLLATE_PAPIPROFILES_H

void vftr_collate_papiprofiles (collated_stacktree_t *collstacktree_ptr,
			        stacktree_t *stacktree_ptr,
				int myrank, int nranks, int *nremote_profiles);

#endif
