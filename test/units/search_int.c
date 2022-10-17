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
   int list[LEN] = {-16965, -16218, -15511, -15407, -14715, -14189, -13694,
                    -12465, -11387, -11204, -9748, -7620, -5997, -5201, -2794,
                    -2402, -1082, 0, 542, 1225, 1708, 2117, 3979, 4393, 6600,
                    7657, 8068, 8547, 10118, 10675, 11975, 13352};

   printf("List to be searched:\n");
   for (int i=0; i<len; i++) {
      printf("%3d %6d\n", i, list[i]);
   }
   printf("\n");

#define NCHECKS 10
   const int nchecks = NCHECKS;
   int checks_match[LEN] = {542, -16218, 13352, -16965, 3979,
                            0, -12465, 7657, -14189, 6600};
   int checks_nomatch[LEN] = {-15241, -11088, -10380, -9445, -7267,
                              1394, 4006, 5487, 5761, 13910};

   printf("Check matching values:\n");
   for (int icheck=0; icheck<nchecks; icheck++) {
      int value = checks_match[icheck];
      int index = vftr_binary_search_int(len, list, value);
      if (index < 0) {
         printf("Error: Value %d could not be found in the list, "
                "although it should.\n", value);
         return 1;
      }
      if (value != list[index]) {
         printf("Error: Value %d was found at index %d in list, "
                "but list[%d] yields %d.\n",
                value, index, index, list[index]);
         return 1;
      }
      printf("Value %d found at index %d\n", value, index);
   }
   printf("\n");

   printf("Check not matching values:\n");
   for (int icheck=0; icheck<nchecks; icheck++) {
      int value = checks_nomatch[icheck];
      int index = vftr_binary_search_int(len, list, value);
      if (index >= 0) {
         printf("Error: Value %d was found in list at index %d.\n", value, index);
         return 1;
      }
      printf("Value %d not found in list\n", value);
   }

#ifdef _MPI
   PMPI_Finalize();
#endif

   return 0;
}
