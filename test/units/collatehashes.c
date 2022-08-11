#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "hashing.h"
#include "collated_hash_types.h"
#include "collate_hashes.h"

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
                                  name, name, function, addrs+1, false);
   name = vftr_get_name_from_address(symboltable, addrs+2);
   int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                  name, name, function, addrs+2, false);
   name = vftr_get_name_from_address(symboltable, addrs+3);
   int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+3, false);
   name = vftr_get_name_from_address(symboltable, addrs+4);
   int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                  name, name, function, addrs+4, false);
   name = vftr_get_name_from_address(symboltable, addrs+5);
   int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                  name, name, function, addrs+5, false);

   vftr_compute_stack_hashes(&stacktree);
   hashlist_t hashlist = vftr_collate_hashes(&stacktree);

   for (int istack=0; istack<hashlist.nhashes; istack++) {
      printf("%d: %016lx\n", istack, hashlist.hashes[istack]);
   }

   free(hashlist.hashes);
   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);

#ifdef _MPI
   PMPI_Finalize();
#endif

   return 0;
}
