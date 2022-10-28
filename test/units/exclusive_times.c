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

   for (int ithread=0; ithread<6; ithread++) {
      vftr_register_dummy_stack (&stacktree, "func0<init", ithread, 10, 0);
   }

   for (int ithread=0; ithread<3; ithread++) {
      vftr_register_dummy_stack (&stacktree, "func1<init", ithread, 10, 0);
   }

   vftr_register_dummy_stack(&stacktree, "func2<func1<init", 1, 3, 0);
   vftr_register_dummy_stack(&stacktree, "func2<func1<init", 2, 4, 0);

   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 0, 3, 0);
   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 2, 2, 0);
   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 4, 3, 0);

   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 1, 4, 0);
   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 3, 5, 0);
   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 5, 6, 0);

   vftr_register_dummy_stack(&stacktree, "func5<func4<func0<init", 5, 2, 0);

   vftr_update_stacks_exclusive_time(&stacktree);

   vftr_print_stacktree(stdout, stacktree);
   fprintf(stdout, "\n");
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      char *stackstr = vftr_get_stack_string(stacktree, istack, false);
      fprintf(stdout, "%d: %s\n", istack, stackstr);
      free(stackstr);
      int nprofs=stacktree.stacks[istack].profiling.nprofiles;
      for (int ithread=0; ithread<nprofs; ithread++) {
         fprintf(stdout, "   Thread: %d (%d)", ithread,
                 stacktree.stacks[istack].profiling.profiles[ithread].threadID);
         profile_t *profile = stacktree.stacks[istack].profiling.profiles+ithread;
         vftr_print_callprofiling(stdout, profile->callprof);
      }
   }


   vftr_stacktree_free(&stacktree);
#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
