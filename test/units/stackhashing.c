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

   int func1_idx = 0;
   int func2_idx = vftr_new_stack(func1_idx, &stacktree, symboltable, function,
                                  addrs+0, false);
   int func3_idx = vftr_new_stack(func1_idx, &stacktree, symboltable, function,
                                  addrs+1, false);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree, symboltable, function,
                                  addrs+2, false);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree, symboltable, function,
                                  addrs+3, false);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree, symboltable, function,
                                  addrs+4, false);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree, symboltable, function,
                                  addrs+5, false);

   vftr_compute_stack_hashes(stacktree.nstacks, stacktree.stacks);
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      printf("%d: %016lx\n", istack, stacktree.stacks[istack].hash);
   }

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_environment_free(&environment);
#ifdef _MPI
   PMPI_Finalize();
#endif

   return 0;
}
