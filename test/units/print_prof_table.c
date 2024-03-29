#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "configuration.h"
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

   config_t config;
   config = vftr_read_config();

   stacktree_t stacktree = vftr_init_dummy_stacktree(2000000000ll, 0);

   vftr_register_dummy_stack(&stacktree, "func0<init", 0, 1000000000ll, 2000ll); 
   vftr_register_dummy_stack(&stacktree, "func0<init", 1, 1000000000ll, 4000ll); 
   vftr_register_dummy_stack(&stacktree, "func0<init", 2, 1000000000ll, 8000ll); 
   vftr_register_dummy_stack(&stacktree, "func0<init", 3, 1000000000ll, 16000ll); 
   vftr_register_dummy_stack(&stacktree, "func0<init", 4, 1000000000ll, 32000ll); 
   vftr_register_dummy_stack(&stacktree, "func0<init", 5, 1000000000ll, 64000ll); 

   vftr_register_dummy_stack(&stacktree, "func1<init", 0, 1000000000ll, 128000ll);
   vftr_register_dummy_stack(&stacktree, "func1<init", 1, 1000000000ll, 256000ll);
   vftr_register_dummy_stack(&stacktree, "func1<init", 2, 1000000000ll, 515000ll);

   vftr_register_dummy_stack(&stacktree, "func2<func1<init", 1, 300000000ll, 1024000ll);
   vftr_register_dummy_stack(&stacktree, "func2<func1<init", 2, 400000000ll, 2048000ll);

   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 0, 300000000ll, 409600);
   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 2, 200000000ll, 8192000ll);
   vftr_register_dummy_stack(&stacktree, "func3<func0<init", 4, 300000000ll, 16384000ll);

   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 1, 400000000ll,3276800);
   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 3, 500000000ll, 65536000ll);
   vftr_register_dummy_stack(&stacktree, "func4<func0<init", 5, 600000000ll, 131072000ll);

   vftr_register_dummy_stack(&stacktree, "func5<func4<func0<init", 5, 200000000ll,262144000ll);

   vftr_update_stacks_exclusive_time(&stacktree);

   // collate stacks to get the global ID
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles(&collated_stacktree, &stacktree);

   vftr_write_ranklogfile_profile_table(stdout, stacktree, config);
   vftr_write_logfile_profile_table(stdout, collated_stacktree, config);

   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_config_free(&config);
#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
