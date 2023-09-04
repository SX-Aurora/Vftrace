#include <stdlib.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <vftrace.h>

#define N_ITER 10


void __attribute__ ((noinline)) do_region () {
   vftrace_region_begin ("user-region-1");
   vftrace_region_end ("user-region-1");
}

int main () {

#ifdef _MPI
   MPI_Init (NULL, NULL);
#endif

   for (int i = 0; i < N_ITER; i++) {
      do_region();
   }

#ifdef _MPI
   MPI_Finalize();
#endif
}
