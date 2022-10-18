#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "environment_types.h"
#include "environment.h"
#include "symbol_types.h"
#include "symbols.h"
#include "stack_types.h"
#include "stacks.h"
#include "hashing.h"
#include "collated_hash_types.h"
#include "collate_hashes.h"

#include "dummy_stacktree.h"
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

   // build stacktree
   stacktree_t stacktree = vftr_init_dummy_stacktree(0ll, 0ll);

   char *name;
   if (myrank == 0) {
      vftr_register_dummy_stack(&stacktree, "func0<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func3<func0<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func4<func0<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func5<func4<func0<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func1<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func2<func1<init", 0, 0ll, 0ll);
   } else {
      vftr_register_dummy_stack(&stacktree, "func0<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func2<func0<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func1<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func2<func1<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func4<func2<func0<init", 0, 0ll, 0ll);
      vftr_register_dummy_stack(&stacktree, "func5<func1<init", 0, 0ll, 0ll);
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
   vftr_stacktree_free(&stacktree);

   PMPI_Finalize();

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
