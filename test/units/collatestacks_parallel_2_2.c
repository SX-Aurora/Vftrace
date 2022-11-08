#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"

#include "dummy_stacktree.h"
#include <mpi.h>


int main(int argc, char **argv) {
   INIT_SELF_PROF_VFTRACE;
   PMPI_Init(&argc, &argv);

   int nranks;
   PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
   if (nranks != 2) {
      fprintf(stderr, "This test requires exacly two processes, "
              "but was started with %d\n", nranks);
      return 1;
   }

   int myrank;
   PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // build stacktree
   stacktree_t stacktree = vftr_init_dummy_stacktree(0ll, 0ll);

   vftr_register_dummy_stack(&stacktree, "func0<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func2<func0<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func1<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func2<func1<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func4<func2<func0<init", 0, 0ll, 0ll);
   vftr_register_dummy_stack(&stacktree, "func5<func1<init", 0, 0ll, 0ll);

   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);

   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);

   PMPI_Finalize();

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
