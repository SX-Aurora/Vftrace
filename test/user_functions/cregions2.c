#ifdef _MPI
#include <mpi.h>
#endif

#include <vftrace.h>

int main() {

#ifdef _MPI
   MPI_Init(NULL, NULL);
#endif

   char *reg_name = "user-region-1";
   vftrace_region_begin(reg_name);
   vftrace_region_end(reg_name);

#ifdef _MPI
   MPI_Finalize();
#endif
}

