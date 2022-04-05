#ifndef COLLATE_HASHES_H
#define COLLATE_HASHES_H

#include <stdlib.h>

#include "stack_types.h"
#include "collated_hash_types.h"

hashlist_t vftr_collate_hashes(stacktree_t *stacktree_ptr);

void vftr_collated_hashlist_free(hashlist_t *hashlist);

#endif
