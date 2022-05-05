#include <stdlib.h>
#include <stdio.h>

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
   if (myrank == 0) {
      int func1_idx = 0;
      name = vftr_get_name_from_address(symboltable, addrs+0);
      int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                     name, function, addrs+0, false);
      name = vftr_get_name_from_address(symboltable, addrs+1);
      int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                     name, function, addrs+1, false);
      name = vftr_get_name_from_address(symboltable, addrs+2);
      int func4_idx = vftr_new_stack(func3_idx, &stacktree,
                                     name, function, addrs+2, false);
      name = vftr_get_name_from_address(symboltable, addrs+3);
      int func5_idx = vftr_new_stack(func2_idx, &stacktree,
                                     name, function, addrs+3, false);
      name = vftr_get_name_from_address(symboltable, addrs+4);
      int func6_idx = vftr_new_stack(func2_idx, &stacktree,
                                     name, function, addrs+4, false);
      name = vftr_get_name_from_address(symboltable, addrs+5);
      int func7_idx = vftr_new_stack(func6_idx, &stacktree,
                                     name, function, addrs+5, false);
   } else {
      int func1_idx = 0;
      name = vftr_get_name_from_address(symboltable, addrs+0);
      int func2_idx = vftr_new_stack(func1_idx, &stacktree,
                                     name, function, addrs+0, false);
      name = vftr_get_name_from_address(symboltable, addrs+1);
      int func3_idx = vftr_new_stack(func1_idx, &stacktree,
                                     name, function, addrs+1, false);
      name = vftr_get_name_from_address(symboltable, addrs+2);
      int func4_idx = vftr_new_stack(func2_idx, &stacktree,
                                     name, function, addrs+2, false);
      name = vftr_get_name_from_address(symboltable, addrs+2);
      int func5_idx = vftr_new_stack(func3_idx, &stacktree,
                                     name, function, addrs+2, false);
      name = vftr_get_name_from_address(symboltable, addrs+4);
      int func6_idx = vftr_new_stack(func4_idx, &stacktree,
                                     name, function, addrs+4, false);
      name = vftr_get_name_from_address(symboltable, addrs+5);
      int func7_idx = vftr_new_stack(func3_idx, &stacktree,
                                     name, function, addrs+5, false);
   }

   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);

#define FILENAME_BUFF_LEN 64
   char filename[FILENAME_BUFF_LEN];
   snprintf(filename, FILENAME_BUFF_LEN*sizeof(char),
            "collatestacks_parallel_3_p%d.tmpout", myrank);
   FILE *fp = fopen(filename, "w");
   vftr_print_collated_stacklist(fp, collated_stacktree);
   fprintf(fp,"\n");
   fclose(fp);
#undef FILENAME_BUFF_LEN

   free_dummy_symbol_table(&symboltable);
   vftr_stacktree_free(&stacktree);
   vftr_collated_stacktree_free(&collated_stacktree);
   vftr_environment_free(&environment);

   PMPI_Finalize();

   return 0;
}
