#include <stdlib.h>
#include <stdio.h>

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

#include "dummysymboltable.h"

#ifdef _MPI
#include <mpi.h>
#endif

int main(int argc, char **argv) {
#if defined(_MPI)
   PMPI_Init(&argc, &argv);
#else
   (void) argc;
   (void) argv;
#endif

   environment_t environment;
   environment = vftr_read_environment();

   // dummy symboltable
   uintptr_t addrs = 123456;
   symboltable_t symboltable = dummy_symbol_table(6, addrs);

   // build stacktree
   stacktree_t stacktree = vftr_new_stacktree();
   // 0: init
   int iprof = 0;
   profile_t *profile = stacktree.stacks[0].profiling.profiles+0;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 20);

   char *name;
   int func1_idx = 0;
   // 1: func0<init
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, function, addrs+0, false);
   for (int ithread=0; ithread<6; ithread++) {
      iprof = vftr_new_profile(ithread,&(stacktree.stacks[func2_idx].profiling));
      profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
      vftr_accumulate_callprofiling(&(profile->callProf), 1, 10);
   }

   // 2: func1<init
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, function, addrs+1, false);
   for (int ithread=0; ithread<3; ithread++) {
      iprof = vftr_new_profile(ithread,&(stacktree.stacks[func3_idx].profiling));
      profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
      vftr_accumulate_callprofiling(&(profile->callProf), 1, 10);
   }

   // 3: func2<func1<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, function, addrs+2, false);
   iprof = vftr_new_profile(1,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 3);
   iprof = vftr_new_profile(2,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 4);

   // 4: func3<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, function, addrs+3, false);
   iprof = vftr_new_profile(0,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 3);
   iprof = vftr_new_profile(2,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 2);
   iprof = vftr_new_profile(4,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 3);

   // 5: func4<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, function, addrs+4, false);
   iprof = vftr_new_profile(1,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 4);
   iprof = vftr_new_profile(3,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 5);
   iprof = vftr_new_profile(5,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 6);

   // 6: func5<func4<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                  name, function, addrs+5, false);
   iprof = vftr_new_profile(5,&(stacktree.stacks[func7_idx].profiling));
   profile = stacktree.stacks[func7_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 2);

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
         vftr_print_callprofiling(stdout, profile->callProf);
      }
   }


   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_environment_free(&environment);
#ifdef _MPI
   PMPI_Finalize();
#endif

   return 0;
}
