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
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "logfile_prof_table.h"

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
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 2000000);

   char *name;
   int func1_idx = 0;
   // 1: func0<init
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+0, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 2ll);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 4ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 8ll);
   iprof = vftr_new_profile_in_list(3,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 16ll);
   iprof = vftr_new_profile_in_list(4,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 32ll);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 64ll);

   // 2: func1<init
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+1, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 128ll);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 256ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1000000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 515ll);

   // 3: func2<func1<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, function, addrs+2, false);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 300000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 1024ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 400000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 2048ll);

   // 4: func3<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+3, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 300000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 4096);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 200000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 8192);
   iprof = vftr_new_profile_in_list(4,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 300000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 16384);

   // 5: func4<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+4, false);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 400000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 32768);
   iprof = vftr_new_profile_in_list(3,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 500000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 65536);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 600000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 131072);

   // 6: func5<func4<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                  name, name, function, addrs+5, false);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func7_idx].profiling));
   profile = stacktree.stacks[func7_idx].profiling.profiles+iprof;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 200000);
   vftr_accumulate_callprofiling_overhead(&(profile->callProf), 262144);

   vftr_update_stacks_exclusive_time(&stacktree);
   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);

   vftr_write_logfile_profile_table(stdout, stacktree, environment);

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_environment_free(&environment);
#ifdef _MPI
   PMPI_Finalize();
#endif

   return 0;
}
