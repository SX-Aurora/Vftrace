#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <collate_stacks.h>
#include <search.h>

int main(int argc, char **argv) {

#ifdef _MPI
   PMPI_Init(&argc, &argv);
#endif

#define LEN 15
   const int len = LEN;
   char *list[LEN] = {"Mandelbrot", "O1-primefactorization", "Testfunction",
                      "Testfunction1", "binary_search", "break-AES",
                      "compute_nth_prime", "find-euler-brick", "fourier",
                      "init", "main", "quicksort", "testfnction",
                      "testfunction", "travelingSalesman"};

   // prepare the dummy collated stacktree
   collated_stacktree_t stacktree = vftr_new_empty_collated_stacktree();
   stacktree.nstacks = len;
   vftr_collated_stacktree_realloc(&stacktree);
   for (int istack=0; istack<len; istack++) {
      stacktree.stacks[istack].name = strdup(list[istack]);
      stacktree.stacks[istack].gid_list.ngids = 0;
   }

   printf("List to be searched:\n");
   for (int istack=0; istack<len; istack++) {
      printf("%3d \"%s\"\n", istack, 
             stacktree.stacks[istack].name);
   }
   printf("\n");

#define NCHECKS 5
   const int nchecks = NCHECKS;
   char *checks_match[LEN] = {"init", "quicksort", "travelingSalesman",
                              "Testfunction", "testfnction"};
   char *checks_nomatch[LEN] = {"Tstfunkction", "Fourier", "Main",
                                "compute-nth_prime",
                                "Much that once was is lost, "
                                "for none now live who remember it."};

   // Setup collated_stacks.

   printf("Check matching names:\n");
   for (int icheck=0; icheck<nchecks; icheck++) {
      char *name = checks_match[icheck];
      int index = vftr_binary_search_collated_stacks_name(stacktree, name);
      if (index < 0) {
         printf("Error: Name %s could not be found in the list, "
                "although it should.\n", name);
         return 1;
      }
      if (name != list[index]) {
         printf("Error: Name %s was found at index %d in list, "
                "but stacktree[%d] yields %s.\n",
                name, index, index, stacktree.stacks[index].name);
         return 1;
      }
      printf("Name %s found at index %d\n", name, index);
   }
   printf("\n");

   printf("Check not matching names:\n");
   for (int icheck=0; icheck<nchecks; icheck++) {
      char *name = checks_nomatch[icheck];
      int index = vftr_binary_search_collated_stacks_name(stacktree, name);
      if (index >= 0) {
         printf("Error: Name %s was found in list at index %d.\n", name, index);
         return 1;
      }
      printf("Name %s not found in list\n", name);
   }

   vftr_collated_stacktree_free(&stacktree);

#ifdef _MPI
   PMPI_Finalize();
#endif

   return 0;
}
