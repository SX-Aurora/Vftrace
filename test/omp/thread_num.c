#include <stdio.h>
#include <stdbool.h>

#include <omp.h>

#include "threads.h"

int main(int argc, char **argv) {
   int threadnum = vftr_get_thread_num();
   printf("My threadnum = %d\n", threadnum);
   for (int nthreads=1; nthreads<4; nthreads++) {
      printf("\n");
      #pragma omp parallel num_threads(nthreads) private(threadnum)
      {
         #pragma omp for ordered
         for (int ithread=0; ithread<nthreads; ithread++) {
            threadnum = vftr_get_thread_num();
            if (threadnum == ithread) {
               #pragma omp ordered
               printf("My threadnum = %d\n", threadnum);
            }
         }
      }
   }

   return 0;
}
