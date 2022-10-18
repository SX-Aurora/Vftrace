#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "callprofiling_types.h"
#include "callprofiling.h"
#include "collated_callprofiling_types.h"
#include "collated_callprofiling.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "collate_profiles.h"
#include <mpi.h>

#include "dummy_stacktree.h"

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

   stacktree_t stacktree = vftr_init_dummy_stacktree(20ll, 0ll);

   if (myrank == 0) {
      vftr_register_dummy_stack(&stacktree, "func0<init", 0, 10ll, 2ll);
      vftr_register_dummy_stack(&stacktree, "func0<init", 1, 10ll, 4ll);
      vftr_register_dummy_stack(&stacktree, "func0<init", 2, 10ll, 8ll);
      vftr_register_dummy_stack(&stacktree, "func0<init", 3, 10ll, 16ll);
      vftr_register_dummy_stack(&stacktree, "func0<init", 4, 10ll, 32ll);
      vftr_register_dummy_stack(&stacktree, "func0<init", 5, 10ll, 64ll);

      vftr_register_dummy_stack(&stacktree, "func1<init", 0, 10ll, 128ll);
      vftr_register_dummy_stack(&stacktree, "func1<init", 1, 10ll, 256ll);
      vftr_register_dummy_stack(&stacktree, "func1<init", 2, 10ll, 515ll);

      vftr_register_dummy_stack(&stacktree, "func2<func1<init", 1, 3ll, 1024ll);
      vftr_register_dummy_stack(&stacktree, "func2<func1<init", 2, 4ll, 2048ll);

      vftr_register_dummy_stack(&stacktree, "func3<func0<init", 0, 3ll, 4096ll);
      vftr_register_dummy_stack(&stacktree, "func3<func0<init", 2, 2ll, 8192ll);
      vftr_register_dummy_stack(&stacktree, "func3<func0<init", 4, 3ll, 16384ll);

      vftr_register_dummy_stack(&stacktree, "func4<func0<init", 1, 4ll, 32768ll);
      vftr_register_dummy_stack(&stacktree, "func4<func0<init", 3, 5ll, 65536ll);
      vftr_register_dummy_stack(&stacktree, "func4<func0<init", 5, 6ll, 131072ll);

      vftr_register_dummy_stack(&stacktree, "func5<func4<func0<init", 5, 2ll, 262144ll);
   } else {
      vftr_register_dummy_stack(&stacktree, "func0<init", 0, 10ll, 2ll);
      vftr_register_dummy_stack(&stacktree, "func0<init", 1, 10ll, 4ll);

      vftr_register_dummy_stack(&stacktree, "func1<init", 0, 10ll, 128ll);
      vftr_register_dummy_stack(&stacktree, "func1<init", 1, 10ll, 256ll);
      vftr_register_dummy_stack(&stacktree, "func1<init", 2, 10ll, 515ll);

      vftr_register_dummy_stack(&stacktree, "func2<func0<init", 1, 3ll, 1024ll);
      vftr_register_dummy_stack(&stacktree, "func2<func0<init", 2, 4ll, 2048ll);

      vftr_register_dummy_stack(&stacktree, "func2<func1<init", 0, 3ll, 4096ll);
      vftr_register_dummy_stack(&stacktree, "func2<func1<init", 2, 2ll, 8192ll);
      vftr_register_dummy_stack(&stacktree, "func2<func1<init", 4, 3ll, 16384ll);

      vftr_register_dummy_stack(&stacktree, "func4<func2<func0<init", 1, 4ll, 32768ll);
      vftr_register_dummy_stack(&stacktree, "func4<func2<func0<init", 3, 5ll, 65536ll);
      vftr_register_dummy_stack(&stacktree, "func4<func2<func0<init", 5, 6ll, 131072ll);

      vftr_register_dummy_stack(&stacktree, "func5<func1<init", 5, 2ll, 262144ll);
   }
   vftr_update_stacks_exclusive_time(&stacktree);

   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles(&collated_stacktree, &stacktree);

   if (myrank == 0) {
      for (int istack=0; istack<collated_stacktree.nstacks; istack++) {
         char *stackstr = vftr_get_collated_stack_string(collated_stacktree, istack, false);
         fprintf(stdout, "%s:\n   ", stackstr);
         vftr_print_collated_callprofiling(stdout,
            collated_stacktree.stacks[istack].profile.callprof);
         free(stackstr);
      }
   }

   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);

   PMPI_Finalize();

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
