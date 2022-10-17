#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "environment_types.h"
#include "environment.h"
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

   stacktree_t stacktree = vftr_init_dummy_stacktree (20, 0);

   vftr_register_dummy_stack(&stacktree, "func0<init", 0, 10, 2);
   vftr_register_dummy_stack(&stacktree, "func0<init", 1, 10, 4);
   vftr_register_dummy_stack(&stacktree, "func0<init", 2, 10, 8);
   vftr_register_dummy_stack(&stacktree, "func0<init", 3, 10, 16);
   vftr_register_dummy_stack(&stacktree, "func0<init", 4, 10, 32);
   vftr_register_dummy_stack(&stacktree, "func0<init", 5, 10, 64);

   vftr_register_dummy_stack(&stacktree, "func1<init", 0, 10, 128);
   vftr_register_dummy_stack(&stacktree, "func1<init", 1, 10, 256);
   vftr_register_dummy_stack(&stacktree, "func1<init", 2, 10, 515);

   vftr_register_dummy_stack(&stacktree, "func2<func1<init", 1, 3, 1024);
   vftr_register_dummy_stack(&stacktree, "func2<func1<init", 2, 4, 2048);

   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 0, 3, 4096);
   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 2, 2, 8192);
   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 4, 3, 16384);

   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 1, 4, 32768);
   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 3, 5, 65536);
   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 5, 6, 131072);

   vftr_register_dummy_stack(&stacktree, "func5<func4<func0<init", 5, 2, 262144);

   vftr_update_stacks_exclusive_time(&stacktree);

   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles(&collated_stacktree, &stacktree);

   for (int istack=0; istack<collated_stacktree.nstacks; istack++) {
      char *stackstr = vftr_get_collated_stack_string(collated_stacktree, istack, false);
      fprintf(stdout, "%s:\n   ", stackstr);
      vftr_print_calltime_imbalances(stdout,
         collated_stacktree.stacks[istack].profile.callprof);
      free(stackstr);
   }

   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);

#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
