#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <vftr_sorting.h>

bool uint64_list_sorted(int n, uint64_t *list) {
   bool sorted = true;
   for (int i=1; i<n; i++) {
      sorted = sorted && (list[i-1] <= list[i]);
   }
   return sorted;
}

int main(int argc, char **argv) {

#ifdef _MPI
   MPI_Init(&argc, &argv);
#endif

   // require cmd-line argument
   if (argc < 2) {
      printf("./radixsort_uint64 <listsize>\n");
      return 1;
   }

   // allocating send/recv buffer
   int n = atoi(argv[1]);
   if (n < 2) {
      printf("listsize needs to be integer >= 2\n");
      return 1;
   }
   uint64_t *list = (uint64_t*) malloc(n*sizeof(uint64_t));
   list[0] = 0xbbb439e77d8b3c8f;
   for (int i=1; i<n; i++) {
      list[i] = list[i-1]<<1;
      if (i%3 == 0) {
         list[i] += 1;
      }
      if (i%5 == 0) {
         list[i] += 1;
      }
   }

   bool sorted_before = uint64_list_sorted(n, list);
   printf("sorted before: %s\n", sorted_before ? "true" : "false");

   vftr_radixsort_uint64(n, list);

   bool sorted_after = uint64_list_sorted(n, list);
   printf("sorted after: %s\n", sorted_after ? "true" : "false");

   free(list);
   list = NULL;

#ifdef _MPI
   MPI_Finalize();
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

