#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "ompprofiling.h"

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
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 1ll);

   char *name;
   int func1_idx = 0;
   // 1: func0<init
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+0, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 2ll);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 4ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 8ll);
   iprof = vftr_new_profile_in_list(3,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 16ll);
   iprof = vftr_new_profile_in_list(4,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 32ll);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func2_idx].profiling));
   profile = stacktree.stacks[func2_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 64ll);

   // 2: func1<init
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+1, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 128ll);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 256ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func3_idx].profiling));
   profile = stacktree.stacks[func3_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 515ll);

   // 3: func2<func1<init
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, function, addrs+2, false);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 1024ll);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func4_idx].profiling));
   profile = stacktree.stacks[func4_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 2048ll);

   // 4: func3<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+3, false);
   iprof = vftr_new_profile_in_list(0,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 4096);
   iprof = vftr_new_profile_in_list(2,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 8192);
   iprof = vftr_new_profile_in_list(4,&(stacktree.stacks[func5_idx].profiling));
   profile = stacktree.stacks[func5_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 16384);

   // 5: func4<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+4, false);
   iprof = vftr_new_profile_in_list(1,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 32768);
   iprof = vftr_new_profile_in_list(3,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 65536);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func6_idx].profiling));
   profile = stacktree.stacks[func6_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 131072);

   // 6: func5<func4<func0<init
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                  name, name, function, addrs+5, false);
   iprof = vftr_new_profile_in_list(5,&(stacktree.stacks[func7_idx].profiling));
   profile = stacktree.stacks[func7_idx].profiling.profiles+iprof;
   vftr_accumulate_ompprofiling_overhead(&(profile->ompprof), 262144);

   int nthreads = 6;
   long long *omp_overheads_nsec = vftr_get_total_omp_overhead(stacktree, nthreads);
   fprintf(stdout, "   OMP Overheads\n");
   for (int i=0; i<nthreads; i++) {
      fprintf(stdout, "      Thread %d: %lld\n", i, omp_overheads_nsec[i]);
   }
   free(omp_overheads_nsec);

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);

#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
