#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <sorting.h>
#include "bad_rng.h"

bool double_list_sorted(int n, double *list) {
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
      printf("./radixsort_perm_double <listsize>\n");
      return 1;
   }

   // allocating send/recv buffer
   int n = atoi(argv[1]);
   if (n < 2) {
      printf("listsize needs to be integer >= 2\n");
      return 1;
   }
   double *list = (double*) malloc(n*sizeof(double));
   double *list2 = (double*) malloc(n*sizeof(double));
   bool sorted_before = true;
   while (sorted_before) {
      for (int i=0; i<n; i++) {
         list[i] = random_double();
         list2[i] = list[i];
      }
      sorted_before = double_list_sorted(n, list);
   }
   printf("sorted before: %s\n", sorted_before ? "true" : "false");

   int *perm = NULL;
   vftr_radixsort_perm_double(n, list, &perm);

   bool sorted_after = double_list_sorted(n, list);
   printf("sorted after: %s\n", sorted_after ? "true" : "false");

   vftr_apply_perm_double(n, list2, perm);

   bool sorted_other_list = double_list_sorted(n, list2);
   printf("other list sorted: %s\n", sorted_other_list ? "true" : "false");

   free(list);
   free(list2);
   free(perm);
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
   } else if (!sorted_other_list) {
      printf("Other list not properly sorted with permutation\n");
      return 1;
   } else {
      return 0;
   }
}
