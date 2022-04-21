#include <stdio.h>
#include <stdbool.h>

#include <omp.h>

#include "vftrace_state.h"

int main(int argc, char **argv) {
   #pragma omp parallel num_threads(2)
   {
      int nthreads = omp_get_num_threads();
      int mythread = omp_get_thread_num();
      printf("Thread %d of %d\n", mythread, nthreads);
   }
   bool omp_tool_started = vftrace.omp_state.tool_started;
   if (!omp_tool_started) {
      printf("OMPT tool not properly started");
      return 1;
   }
   return 0;
}
