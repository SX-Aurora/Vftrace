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

   stacktree_t stacktree = vftr_init_dummy_stacktree (0, 1);

   vftr_register_dummy_stack (&stacktree, "func0<init", 0, 0, 2);
   vftr_register_dummy_stack (&stacktree, "func0<init", 1, 0, 4);
   vftr_register_dummy_stack (&stacktree, "func0<init", 2, 0, 8);
   vftr_register_dummy_stack (&stacktree, "func0<init", 3, 0, 16);
   vftr_register_dummy_stack (&stacktree, "func0<init", 4, 0, 32);
   vftr_register_dummy_stack (&stacktree, "func0<init", 5, 0, 64);

   vftr_register_dummy_stack (&stacktree, "func1<init", 0, 0, 128);
   vftr_register_dummy_stack (&stacktree, "func1<init", 1, 0, 256);
   vftr_register_dummy_stack (&stacktree, "func1<init", 2, 0, 515);

   vftr_register_dummy_stack (&stacktree, "func2<func1<init", 1, 0, 1024);
   vftr_register_dummy_stack (&stacktree, "func2<func1<init", 2, 0, 2048);

   vftr_register_dummy_stack (&stacktree, "func3<func0<init", 0, 0, 4096);
   vftr_register_dummy_stack (&stacktree, "func3<func0<init", 2, 0, 8192);
   vftr_register_dummy_stack (&stacktree, "func3<func0<init", 4, 0, 16384);

   vftr_register_dummy_stack (&stacktree, "func4<func0<init", 1, 0, 32768);
   vftr_register_dummy_stack (&stacktree, "func4<func0<init", 3, 0, 65536);
   vftr_register_dummy_stack (&stacktree, "func4<func0<init", 5, 0, 131072);

   vftr_register_dummy_stack (&stacktree, "func5<func4<func0<init", 5, 0, 262144);

   int nthreads = 6;
   long long *call_overheads_nsec = vftr_get_total_call_overhead(stacktree, nthreads);
   fprintf(stdout, "   Call Overheads\n");
   for (int i=0; i<nthreads; i++) {
      fprintf(stdout, "      Thread %d: %lld\n", i, call_overheads_nsec[i]);
   }
   free(call_overheads_nsec);

   vftr_stacktree_free(&stacktree);
#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
