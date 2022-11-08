#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "configuration.h"
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
#include "sorting.h"

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

   config_t config;
   config = vftr_read_config();

   stacktree_t stacktree = vftr_init_dummy_stacktree (20, 0);

   vftr_register_dummy_stack (&stacktree, "func0<init", 0, 10, 2); 
   vftr_register_dummy_stack (&stacktree, "func0<init", 1, 10, 4); 
   vftr_register_dummy_stack (&stacktree, "func0<init", 2, 10, 8); 
   vftr_register_dummy_stack (&stacktree, "func0<init", 3, 10, 16); 
   vftr_register_dummy_stack (&stacktree, "func0<init", 4, 10, 32); 
   vftr_register_dummy_stack (&stacktree, "func0<init", 5, 10, 64); 

   vftr_register_dummy_stack (&stacktree, "func1<init", 0, 10, 128);
   vftr_register_dummy_stack (&stacktree, "func1<init", 1, 10, 256);
   vftr_register_dummy_stack (&stacktree, "func1<init", 2, 10, 515);

   vftr_register_dummy_stack (&stacktree, "func2<func1<init", 1, 3, 1024);
   vftr_register_dummy_stack (&stacktree, "func2<func1<init", 2, 4, 2048);

   vftr_register_dummy_stack (&stacktree, "func3<func0<init", 0, 3, 4096);
   vftr_register_dummy_stack (&stacktree, "func3<func0<init", 2, 2, 8192);
   vftr_register_dummy_stack (&stacktree, "func3<func0<init", 4, 3, 16384);

   vftr_register_dummy_stack (&stacktree, "func4<func0<init", 1, 4, 32768);
   vftr_register_dummy_stack (&stacktree, "func4<func0<init", 3, 5, 65536);
   vftr_register_dummy_stack (&stacktree, "func4<func0<init", 5, 6, 131072);

   vftr_register_dummy_stack (&stacktree, "func5<func4<func0<init", 5, 2, 262144);

   vftr_update_stacks_exclusive_time(&stacktree);

   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles(&collated_stacktree, &stacktree);

   collated_stack_t **stackptrs =
      vftr_sort_collated_stacks_for_prof(config, collated_stacktree);

   for (int istack=0; istack<collated_stacktree.nstacks; istack++) {
      collated_stack_t *stack = stackptrs[istack];
      int stackID = stack->gid;
      char *stackstr = vftr_get_collated_stack_string(collated_stacktree, stackID, false);
      fprintf(stdout, "%d: %s\n", stackID, stackstr);
      free(stackstr);
      fprintf(stdout, " Overhead: %8lld, ", stack->profile.callprof.overhead_nsec);
      vftr_print_collated_callprofiling(stdout, stack->profile.callprof);
   }

   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_config_free(&config);
#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
