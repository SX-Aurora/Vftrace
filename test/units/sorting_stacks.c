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
#include "collated_stack_types.h"
#include "collate_stacks.h"
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

   vftr_register_dummy_stack (&stacktree, "func5<func4<func0<init", 5, 2,262144);

   vftr_update_stacks_exclusive_time(&stacktree);

   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);

   vftr_stack_t **stackptrs = vftr_sort_stacks_for_prof(config, stacktree);

   for (int istack=0; istack<stacktree.nstacks; istack++) {
      vftr_stack_t *stack = stackptrs[istack];
      int stackID = stack->lid;
      char *stackstr = vftr_get_stack_string(stacktree, stackID, false);
      fprintf(stdout, "%d(g%d): %s\n", stackID, stack->gid, stackstr);
      free(stackstr);
      int nprofs=stack->profiling.nprofiles;
      for (int ithread=0; ithread<nprofs; ithread++) {
         fprintf(stdout, "   Thread: %d (%d)", ithread,
                 stack->profiling.profiles[ithread].threadID);
         profile_t *profile = stack->profiling.profiles+ithread;
         fprintf(stdout, " Overhead: %8lld, ", profile->callprof.overhead_nsec);
         vftr_print_callprofiling(stdout, profile->callprof);
      }
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
