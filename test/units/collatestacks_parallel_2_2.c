#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "environment_types.h"
#include "environment.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"

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

   environment_t environment;
   environment = vftr_read_environment();

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
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, addrs+1, false);
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, addrs+2, false);
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func5_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, addrs+2, false);
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func4_idx, &stacktree,
                                  name, name, addrs+4, false);
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, addrs+5, false);

   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_environment_free(&environment);

   PMPI_Finalize();

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
