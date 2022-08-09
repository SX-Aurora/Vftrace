#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

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
#include "overheadprofiling_types.h"
#include "overheadprofiling.h"
#include "mpi_state_types.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "logfile_mpi_table.h"
#include "mpiprofiling.h"

#include "dummysymboltable.h"

#include <mpi.h>

int main(int argc, char **argv) {
   PMPI_Init(&argc, &argv);

   environment_t environment;
   environment = vftr_read_environment();

   int ranklist[] = {0,1};
   mpi_state_t mpi_state = {
      .nprof_ranks = 2,
      .prof_ranks = ranklist,
      .my_rank_in_prof=true
   };

   // dummy symboltable
   uintptr_t addrs = 123456;
   symboltable_t symboltable = dummy_symbol_table(6, addrs);

   // build stacktree
   stacktree_t stacktree = vftr_new_stacktree();
   // 0: init
   int iprof = 0;
   profile_t *profile = stacktree.stacks[0].profiling.profiles+0;
   vftr_accumulate_callprofiling(&(profile->callProf), 1, 1);

   char *name;
   int func1_idx = 0;
   // 1: func0<init
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, function, addrs+0, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;


   // 2: func1<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, function, addrs+1, false);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiProf), mpi_state,
                                send, 100, 0, 4, 1, 0,
                                1000000, 2000000);
   vftr_accumulate_message_info(&(profile->mpiProf), mpi_state,
                                send, 50, 0, 4, 1, 0,
                                1000000, 2000000);
   vftr_accumulate_message_info(&(profile->mpiProf), mpi_state,
                                recv, 27, 0, 8, 1, 0,
                                1000000, 3000000);
   iprof = vftr_new_profile_in_list(3,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiProf), mpi_state,
                                send, 137, 0, 4, 1, 0,
                                1000000, 4000000);

   // 3: func2<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, function, addrs+2, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func4_idx].profiling));

   // 4: func3<func2<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func4_idx, &stacktree,
                                  name, function, addrs+3, false);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiProf), mpi_state,
                                send, 42, 0, 4, 1, 0,
                                1000000,9000000);

   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);

   vftr_write_logfile_mpi_table(stdout, stacktree, environment);

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_environment_free(&environment);

   PMPI_Finalize();

   return 0;
}
