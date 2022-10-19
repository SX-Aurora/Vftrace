#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "testall_c2vftr.h"

int MPI_Testall(int count, MPI_Request array_of_requests[],
                int *flag, MPI_Status array_of_statuses[]) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
   } else {
      return vftr_MPI_Testall_c2vftr(count, array_of_requests, flag, array_of_statuses);
   }
}

#endif
