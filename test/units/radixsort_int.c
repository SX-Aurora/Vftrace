#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <sorting.h>
#include "bad_rng.h"

bool int_list_sorted(int n, int *list) {
   bool sorted = true;
   for (int i=1; i<n; i++) {
      sorted = sorted && (list[i-1] <= list[i]);
   }
   return sorted;
}

int main(int argc, char **argv) {

#ifdef _MPI
   PMPI_Init(&argc, &argv);
#endif

   // require cmd-line argument
   if (argc < 2) {
      printf("./radixsort_int <listsize>\n");
      return 1;
   }

   // allocating send/recv buffer
   int n = atoi(argv[1]);
   if (n < 2) {
      printf("listsize needs to be integer >= 2\n");
      return 1;
   }
   int *list = (int*) malloc(n*sizeof(int));
   bool sorted_before = true;
   while (sorted_before) {
      for (int i=0; i<n; i++) {
         list[i] = random_int();
      }
      sorted_before = int_list_sorted(n, list);
   }
   printf("sorted before: %s\n", sorted_before ? "true" : "false");

   vftr_radixsort_int(n, list);

   bool sorted_after = int_list_sorted(n, list);
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
