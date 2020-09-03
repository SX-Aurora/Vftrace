#include <stdlib.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <vftrace.h>

int main() {

#ifdef _MPI
   MPI_Init(NULL, NULL);
#endif

   vftrace_region_begin("user-region-1");
   vftrace_region_end("user-region-1");

#ifdef _MPI
   MPI_Finalize();
#endif
}

