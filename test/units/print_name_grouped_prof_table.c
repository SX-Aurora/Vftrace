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
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "collate_profiles.h"
#include "ranklogfile_prof_table.h"
#include "logfile_prof_table.h"

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

   environment_t environment;
   environment = vftr_read_environment();

   stacktree_t stacktree = vftr_init_dummy_stacktree(65536000000000ll, 0ll);

   vftr_register_dummy_stack (&stacktree, "func0<init",             0, 4096000000000ll, 0ll); 
   vftr_register_dummy_stack (&stacktree, "func2<func0<init",       0, 2048000000000ll, 0ll); 

   vftr_register_dummy_stack (&stacktree, "func1<init",             0, 1024000000000ll, 0ll); 
   vftr_register_dummy_stack (&stacktree, "func2<func1<init",       0,  512000000000ll, 0ll); 
   vftr_register_dummy_stack (&stacktree, "func3<func1<init",       0,  256000000000ll, 0ll); 
   vftr_register_dummy_stack (&stacktree, "func4<func1<init",       0,  128000000000ll, 0ll); 

   vftr_register_dummy_stack (&stacktree, "func5<init",             0,   64000000000ll, 0ll);
   vftr_register_dummy_stack (&stacktree, "func3<func5<init",       0,   32000000000ll, 0ll);
   vftr_register_dummy_stack (&stacktree, "func4<func5<init",       0,   16000000000ll, 0ll);

   vftr_register_dummy_stack (&stacktree, "func2<func4<func5<init", 0,    4000000000ll, 0ll);
   vftr_register_dummy_stack (&stacktree, "func3<func4<func5<init", 0,    2000000000ll, 0ll);

   vftr_update_stacks_exclusive_time(&stacktree);

   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles(&collated_stacktree, &stacktree);

   // namegroup
   collated_stacktree_t namegrouped_collated_tacktree =
      vftr_collated_stacktree_group_by_name(&collated_stacktree);

   //vftr_write_ranklogfile_profile_table(stdout, stacktree, environment);
   vftr_write_logfile_profile_table(stdout, collated_stacktree, environment);
   vftr_write_logfile_name_grouped_profile_table(stdout, namegrouped_collated_tacktree,
                                                 environment);

   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_collated_stacktree_free(&namegrouped_collated_tacktree);
   vftr_environment_free(&environment);

#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
