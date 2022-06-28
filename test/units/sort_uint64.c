#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <sorting.h>
#include "bad_rng.h"

bool uint64_list_sorted(int n, uint64_t *list, bool ascending) {
   bool sorted = true;
   if (ascending) {
      for (int i=1; i<n; i++) {
         sorted = sorted && (list[i-1] <= list[i]);
      }
   } else {
      for (int i=1; i<n; i++) {
         sorted = sorted && (list[i-1] >= list[i]);
      }
   }
   return sorted;
}

int main(int argc, char **argv) {

#ifdef _MPI
   PMPI_Init(&argc, &argv);
#endif

   // require cmd-line argument
   if (argc < 3) {
      printf("./sort_uint64 <listsize> <ascending>\n");
      return 1;
   }

   int n = atoi(argv[1]);
   if (n < 2) {
      printf("listsize needs to be integer >= 2\n");
      return 1;
   }

   int ascending_int = atoi(argv[2]);
   bool ascending = ascending_int ? true : false;

   uint64_t *list = (uint64_t*) malloc(n*sizeof(uint64_t));
   bool sorted_before = true;
   while (sorted_before) {
      for (int i=0; i<n; i++) {
         list[i] = random_uint64();
      }
      sorted_before = uint64_list_sorted(n, list, ascending);
   }
   printf("sorted before: %s\n", sorted_before ? "true" : "false");

   vftr_sort_uint64(n, list, ascending);

   bool sorted_after = uint64_list_sorted(n, list, ascending);
   printf("sorted after: %s\n", sorted_after ? "true" : "false");

   free(list);
   list = NULL;

#ifdef _MPI
   PMPI_Finalize();
#endif

   if (sorted_before) {
      printf("Initial test array was already sorted\n");
      return 1;
   } else if (!sorted_after) {
      printf("Array was not properly sorted\n");
      return 1;
   } else {
      return 0;
   }
}
