#include <stdlib.h>
#include <stdio.h>

#include "environment_types.h"
#include "environment.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "hashing.h"
#include "collated_hash_types.h"
#include "collate_hashes.h"

#include "dummysymboltable.h"
#include <mpi.h>

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);

   int nranks;
   MPI_Comm_size(MPI_COMM_WORLD, &nranks);
   if (nranks != 2) {
      fprintf(stderr, "This test requires exacly two processes, "
              "but was started with %d\n", nranks);
      return 1;
   }

   int myrank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // dummy symboltable
   uintptr_t addrs = 123456;
   symboltable_t symboltable = dummy_symbol_table(6, addrs);

   // build stacktree
   stacktree_t stacktree = vftr_new_stacktree();

   if (myrank == 0) {
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
   } else {
      int func1_idx = 0;
      int func2_idx = vftr_new_stack(func1_idx, &stacktree, symboltable, function,
                                     addrs+0, false);
      int func3_idx = vftr_new_stack(func1_idx, &stacktree, symboltable, function,
                                     addrs+1, false);
      int func4_idx = vftr_new_stack(func2_idx, &stacktree, symboltable, function,
                                     addrs+2, false);
      int func5_idx = vftr_new_stack(func3_idx, &stacktree, symboltable, function,
                                     addrs+2, false);
      int func6_idx = vftr_new_stack(func4_idx, &stacktree, symboltable, function,
                                     addrs+4, false);
      int func7_idx = vftr_new_stack(func3_idx, &stacktree, symboltable, function,
                                     addrs+5, false);
   }

   vftr_compute_stack_hashes(&stacktree);
   hashlist_t hashlist = vftr_collate_hashes(&stacktree);

#define FILENAME_BUFF_LEN 64
   char filename[FILENAME_BUFF_LEN];
   snprintf(filename, FILENAME_BUFF_LEN*sizeof(char),
            "collatehashes_parallel_p%d.tmpout", myrank);
   FILE *fp = fopen(filename, "w");
   for (int istack=0; istack<hashlist.nhashes; istack++) {
      fprintf(fp, "%d: %016lx\n", istack, hashlist.hashes[istack]);
   }
   fprintf(fp, "\n");
   fclose(fp);
#undef FILENAME_BUFF_LEN

   free(hashlist.hashes);
   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);

   PMPI_Finalize();

   return 0;
}
