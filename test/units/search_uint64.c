#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <search.h>

int main(int argc, char **argv) {

#ifdef _MPI
   PMPI_Init(&argc, &argv);
#endif

#define LEN 32
   const int len = LEN;
   uint64_t list[LEN] = {0, 5417, 6840, 8368, 8383, 8783, 10993, 13138, 13665,
                         14278, 14658, 14801, 15945, 16192, 17930, 18990, 19066,
                         19090, 19469, 20743, 21117, 21791, 24260, 25550, 25596,
                         25908, 26892, 28319, 28711, 29628, 31645, 32490};

   printf("List to be searched:\n");
   for (int i=0; i<len; i++) {
      printf("%3d %6ld\n", i, list[i]);
   }
   printf("\n");

#define NCHECKS 10
   const int nchecks = NCHECKS;
   uint64_t checks_match[LEN] = {32490, 20743, 0, 8783, 26892,
                                 13665, 19090, 18990, 25596, 14801};
   uint64_t checks_nomatch[LEN] = {64123, 17, 22000, 20877, 13666,
                                   29065, 6402, 26174, 13444, 5561};

   printf("Check matching values:\n");
   for (int icheck=0; icheck<nchecks; icheck++) {
      uint64_t value = checks_match[icheck];
      int index = vftr_binary_search_uint64(len, list, value);
      if (index < 0) {
         printf("Error: Value %ld could not be found in the list, "
                "although it should.\n", value);
         return 1;
      }
      if (value != list[index]) {
         printf("Error: Value %ld was found at index %d in list, "
                "but list[%d] yields %ld.\n",
                value, index, index, list[index]);
         return 1;
      }
      printf("Value %ld found at index %d\n", value, index);
   }
   printf("\n");

   printf("Check not matching values:\n");
   for (int icheck=0; icheck<nchecks; icheck++) {
      uint64_t value = checks_nomatch[icheck];
      int index = vftr_binary_search_uint64(len, list, value);
      if (index >= 0) {
         printf("Error: Value %ld was found in list at index %d.\n", value, index);
         return 1;
      }
      printf("Value %ld not found in list\n", value);
   }

#ifdef _MPI
   PMPI_Finalize();
#endif

   return 0;
}
