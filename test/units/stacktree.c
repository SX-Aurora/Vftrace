#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling.h"

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

   char *name;
   int func1_idx = 0;
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+0, false);
   vftr_new_profile_in_list(0,&(stacktree.stacks[func2_idx].profiling));

   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+1, false);
   vftr_new_profile_in_list(0,&(stacktree.stacks[func3_idx].profiling));

   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, addrs+2, false);
   vftr_new_profile_in_list(0,&(stacktree.stacks[func4_idx].profiling));

   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, addrs+3, false);
   vftr_new_profile_in_list(0,&(stacktree.stacks[func5_idx].profiling));

   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, addrs+4, false);
   vftr_new_profile_in_list(0,&(stacktree.stacks[func6_idx].profiling));

   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                  name, name, addrs+5, false);
   vftr_new_profile_in_list(0,&(stacktree.stacks[func7_idx].profiling));

   vftr_print_stacktree(stdout, stacktree);

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
