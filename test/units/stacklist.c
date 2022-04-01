#include <stdlib.h>
#include <stdio.h>

#include "environment_types.h"
#include "environment.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"

#ifdef _MPI
#include <mpi.h>
#endif

void fill_symbol_table(symboltable_t *symboltable_ptr, uintptr_t baseaddr) {
   for (unsigned int isym=0; isym<symboltable_ptr->nsymbols; isym++) {
      symboltable_ptr->symbols[isym].addr = baseaddr+isym;
      symboltable_ptr->symbols[isym].index = 0;
   }
   symboltable_ptr->symbols[0].name = "func0";
   symboltable_ptr->symbols[1].name = "func1";
   symboltable_ptr->symbols[2].name = "func2";
   symboltable_ptr->symbols[3].name = "func3";
   symboltable_ptr->symbols[4].name = "func4";
   symboltable_ptr->symbols[5].name = "func5";
}

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
   symboltable_t symboltable = {.nsymbols=6, .symbols=NULL};
   symboltable.symbols = (symbol_t*) malloc(symboltable.nsymbols*sizeof(symbol_t));
   fill_symbol_table(&symboltable, addrs);

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

   vftr_print_stacklist(stdout, stacktree);

   free(symboltable.symbols);
   vftr_stacktree_free(&stacktree);
   vftr_environment_free(&environment);
#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
