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
#include "mpiprofiling_types.h"
#include "mpiprofiling.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "collate_profiles.h"
#include "mpi_util_types.h"

#include "dummysymboltable.h"

#include <mpi.h>

int main(int argc, char **argv) {
   INIT_SELF_PROF_VFTRACE;
   PMPI_Init(&argc, &argv);

   int nranks;
   PMPI_Comm_size(MPI_COMM_WORLD, &nranks);
   if (nranks != 2) {
      fprintf(stderr, "This test requires exacly two processes, "
              "but was started with %d\n", nranks);
      return 1;
   }

   int myrank;
   PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // dummy mpi_state
   int rankarray[] = {0,1};
   mpi_state_t mpi_state = {
      .pcontrol_level = 1,
      .nopen_requests = 0,
      .open_requests = NULL,
      .nprof_ranks = 2,
      .prof_ranks = rankarray,
      .my_rank_in_prof = true
   };

   // dummy symboltable
   uintptr_t addrs = 123456;
   symboltable_t symboltable = dummy_symbol_table(6, addrs);

   // build stacktree
   stacktree_t stacktree = vftr_new_stacktree();
   // 0: init
   int iprof = 0;
   profile_t *profile = stacktree.stacks[0].profiling.profiles+0;

   char *name;
   int func1_idx = 0;
   // 1: func0<init
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+0, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, send,
                                2, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, recv,
                                4, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, send,
                                8, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, recv,
                                16, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, send,
                                32, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, recv,
                                64, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   
   // 2: func1<init
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+1, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, send,
                                128, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, recv,
                                256, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   
   // 3: func2<func1<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, function, addrs+2, false);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, send,
                                512, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, recv,
                                1024, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, send,
                                2048, 0, (int) sizeof(char),
                                1, 0, 0, 1);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state, recv,
                                4096, 0, (int) sizeof(char),
                                1, 0, 0, 1);

   vftr_update_stacks_exclusive_time(&stacktree);

   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles(&collated_stacktree, &stacktree);

   if (myrank == 0) {
      for (int istack=0; istack<collated_stacktree.nstacks; istack++) {
         char *stackstr = vftr_get_collated_stack_string(collated_stacktree, istack, false);
         fprintf(stdout, "%s:\n   ", stackstr);
         vftr_print_mpiprofiling(stdout,
            collated_stacktree.stacks[istack].profile.mpiprof);
         free(stackstr);
      }
   }

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);

   PMPI_Finalize();

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
