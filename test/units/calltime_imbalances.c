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

#include "dummysymboltable.h"

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

   // dummy symboltable
   uintptr_t addrs = 123456;
   symboltable_t symboltable = dummy_symbol_table(6, addrs);

   // build stacktree
   stacktree_t stacktree = vftr_new_stacktree();
   // 0: init
   int iprof = 0;
   profile_t *profile = stacktree.stacks[0].profiling.profiles+0;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 20);

   char *name;
   int func1_idx = 0;
   // 1: func0<init
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+0, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 2ll);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 4ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 8ll);
   iprof = vftr_new_profile_in_list(3,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 16ll);
   iprof = vftr_new_profile_in_list(4,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 32ll);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 64ll);

   // 2: func1<init
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+1, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 128ll);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 256ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 10);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 515ll);

   // 3: func2<func1<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, function, addrs+2, false);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 3);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 1024ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 4);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 2048ll);

   // 4: func3<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+3, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 3);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 4096);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 2);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 8192);
   iprof = vftr_new_profile_in_list(4,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 3);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 16384);

   // 5: func4<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+4, false);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 4);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 32768);
   iprof = vftr_new_profile_in_list(3,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 5);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 65536);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 6);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 131072);

   // 6: func5<func4<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                  name, name, function, addrs+5, false);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func7_idx].profiling));
   profile = stacktree.stacks[func7_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callprof), 1, 2);
   vftr_accumulate_callprofiling_overhead(&(profile->callprof), 262144);

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

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);

#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
