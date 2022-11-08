#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "configuration.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "mpi_state_types.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "collate_profiles.h"
#include "ranklogfile_mpi_table.h"
#include "logfile_mpi_table.h"
#include "mpiprofiling.h"

#include "dummysymboltable.h"

#include <mpi.h>

int main(int argc, char **argv) {
   INIT_SELF_PROF_VFTRACE;
   PMPI_Init(&argc, &argv);

   config_t config;
   config = vftr_read_config();

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

   char *name;
   int func1_idx = 0;
   // 1: func0<init
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+0, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 662, 0, 4, 1, 0,
                                1000000000ll, 5809000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 385, 0, 4, 1, 0,
                                1000000000ll, 9943000000ll);


   // 2: func1<init
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+1, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 962, 0, 4, 1, 0,
                                1000000000ll, 5681000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 448, 0, 4, 1, 0,
                                1000000000ll, 1112000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 924, 0, 4, 1, 0,
                                1000000000ll, 1254000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 750, 0, 4, 1, 0,
                                1000000000ll, 5895000000ll);

   
   // 3: func2<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+2, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 688, 0, 4, 1, 0,
                                1000000000ll, 9288000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 892, 0, 4, 1, 0,
                                1000000000ll, 2438000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 148, 0, 4, 1, 0,
                                1000000000ll, 1244000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 960, 0, 4, 1, 0,
                                1000000000ll, 9191000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 857, 0, 4, 1, 0,
                                1000000000ll, 5601000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 538, 0, 4, 1, 0,
                                1000000000ll, 2126000000ll);

   // 4: func3<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+3, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 124, 0, 4, 1, 0,
                                1000000000ll, 1175000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 969, 0, 4, 1, 0,
                                1000000000ll, 8940000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 165, 0, 4, 1, 0,
                                1000000000ll, 3656000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 299, 0, 4, 1, 0,
                                1000000000ll, 8926000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 657, 0, 4, 1, 0,
                                1000000000ll, 5032000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 858, 0, 4, 1, 0,
                                1000000000ll, 8336000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 597, 0, 4, 1, 0,
                                1000000000ll, 2787000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 701, 0, 4, 1, 0,
                                1000000000ll, 9338000000ll);

   // 5: func4<init
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+4, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 978, 0, 4, 1, 0,
                                1000000000ll, 7145000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 783, 0, 4, 1, 0,
                                1000000000ll, 3260000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 932, 0, 4, 1, 0,
                                1000000000ll, 3030000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 755, 0, 4, 1, 0,
                                1000000000ll, 9511000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 848, 0, 4, 1, 0,
                                1000000000ll, 3995000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 253, 0, 4, 1, 0,
                                1000000000ll, 2281000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 748, 0, 4, 1, 0,
                                1000000000ll, 1315000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 284, 0, 4, 1, 0,
                                1000000000ll, 1500000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 958, 0, 4, 1, 0,
                                1000000000ll, 9463000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 886, 0, 4, 1, 0,
                                1000000000ll, 5165000000ll);

   // 6: func5<init
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+5, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func7_idx].profiling));
   profile = stacktree.stacks[func7_idx].profiling.profiles+iprof;
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 146, 0, 4, 1, 0,
                                1000000000ll, 5342000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 712, 0, 4, 1, 0,
                                1000000000ll, 3268000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 732, 0, 4, 1, 0,
                                1000000000ll, 6521000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 394, 0, 4, 1, 0,
                                1000000000ll, 2201000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 324, 0, 4, 1, 0,
                                1000000000ll, 6562000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 768, 0, 4, 1, 0,
                                1000000000ll, 7521000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 409, 0, 4, 1, 0,
                                1000000000ll, 2351000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 481, 0, 4, 1, 0,
                                1000000000ll, 9087000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 581, 0, 4, 1, 0,
                                1000000000ll, 7992000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 465, 0, 4, 1, 0,
                                1000000000ll, 3290000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                send, 155, 0, 4, 1, 0,
                                1000000000ll, 6654000000ll);
   vftr_accumulate_message_info(&(profile->mpiprof), mpi_state,
                                recv, 544, 0, 4, 1, 0,
                                1000000000ll, 5369000000ll);

   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles(&collated_stacktree, &stacktree);

   vftr_write_ranklogfile_mpi_table(stdout, stacktree, config);
   vftr_write_logfile_mpi_table(stdout, collated_stacktree, config);

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_config_free(&config);

   PMPI_Finalize();

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
