#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "environment_types.h"
#include "environment.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "callprofiling_types.h"
#include "callprofiling.h"

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

   environment_t environment;
   environment = vftr_read_environment();

   // build stacktree
   stacktree_t stacktree = vftr_new_stacktree();
   int nthreads = 10;
   for (int ithread=0; ithread<nthreads; ithread++) {
      vftr_new_profile_in_list(ithread, &(stacktree.stacks[0].profiling));
      profile_t *profile = stacktree.stacks[0].profiling.profiles+ithread;
      vftr_accumulate_callprofiling(&(profile->callprof),
                                    ithread*ithread, // calls
                                    137*ithread);
   }

   for (int ithread=0; ithread<nthreads; ithread++) {
      fprintf(stdout, "Thread: %d ", ithread);
      profile_t *profile = stacktree.stacks[0].profiling.profiles+ithread;
      vftr_print_callprofiling(stdout, profile->callprof);
   }

   vftr_stacktree_free(&stacktree);
   vftr_environment_free(&environment);
#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
