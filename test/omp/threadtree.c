#include <stdio.h>
#include <stdbool.h>

#include <omp.h>

#include "threads.h"
#include "vftrace_state.h"

void fnk1(void) {
   #pragma omp parallel num_threads(2)
   {
      int threadnum = vftr_get_thread_num();
      if (threadnum == 0) {
         vftr_print_threadtree(stdout, vftrace.process.threadtree);
      }
   }
}

int main(int argc, char **argv) {
   #pragma omp parallel num_threads(2)
   {
      int threadnum = vftr_get_thread_num();
      if (threadnum == 0) {
         fnk1();
      }
   }

   return 0;
}
