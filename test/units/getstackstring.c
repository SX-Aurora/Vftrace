#include <stdlib.h>
#include <stdio.h>

#include "environment_types.h"
#include "environment.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"

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

   char *name;
   int func1_idx = 0;
   name = vftr_get_name_from_address(symboltable, addrs+0);
   int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+0, false);
   name = vftr_get_name_from_address(symboltable, addrs+1);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                  name, name, function, addrs+1, true);
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, function, addrs+2, false);
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+3, true);
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+4, false);
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                  name, name, function, addrs+5, true);

   for (int istack=0; istack<stacktree.nstacks; istack++) {
      char *stackstr = vftr_get_stack_string(stacktree, istack, false);
      printf("%d: %s\n", istack, stackstr);
      free(stackstr);
   }

   printf("\n");

   for (int istack=0; istack<stacktree.nstacks; istack++) {
      char *stackstr = vftr_get_stack_string(stacktree, istack, true);
      printf("%d: %s\n", istack, stackstr);
      free(stackstr);
   }

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_environment_free(&environment);
#ifdef _MPI
   PMPI_Finalize();
#endif

   return 0;
}
