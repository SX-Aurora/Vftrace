#include <stdlib.h>

#include "stack_types.h"

#include "hashing.h"
#include "collate_hashes.h"

void vftr_collate_stacks(stacktree_t *stacktree_ptr) {
   // first compute the hashes for all stacks
   vftr_compute_stack_hashes(stacktree_ptr);

   // collate hashes between processes
   vftr_collate_hashes(stacktree_ptr);
   
}

void vftr_collated_stacktree_free(int *nstacks_ptr,
                                  collated_stack_t **stacks_ptr) {
   int nstacks = *nstacks_ptr;
   collated_stack_t *stacks = *stacks_ptr;
//   if (nstacks > 0) {
//      for (int istack=0; istack<nstacks; istack++) {
//         free(stacks[istack].name);
//      }
//      free(stacks);
//      stacks = NULL;
//      nstacks = 0;
//      *stacks_ptr = stacks;
//      *nstacks_ptr = nstacks;
//   }
}
