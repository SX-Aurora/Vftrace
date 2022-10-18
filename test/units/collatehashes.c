#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "hashing.h"
#include "collated_hash_types.h"
#include "collate_hashes.h"

#include "dummy_stacktree.h"

#ifdef _MPI
#include <mpi.h>
#endif

int main(int argc, char **argv) {
   INIT_SELF_PROF_VFTRACE;
#if defined(_MPI)
   PMPI_Init(&argc, &argv);
#else
   (void) argc;
   (void) argv;
#endif

   // build stacktree
   stacktree_t stacktree = vftr_init_dummy_stacktree(0ll, 0ll);

   vftr_register_dummy_stack(&stacktree, "func0<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func1<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func2<func1<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func5<func4<func0<init", 0, 0ll, 0ll);

   vftr_compute_stack_hashes(&stacktree);
   hashlist_t hashlist = vftr_collate_hashes(&stacktree);

   for (int istack=0; istack<hashlist.nhashes; istack++) {
      printf("%d: %016lx\n", istack, hashlist.hashes[istack]);
   }

   free(hashlist.hashes);
   vftr_stacktree_free(&stacktree);

#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
