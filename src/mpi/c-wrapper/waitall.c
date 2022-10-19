#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "waitall_c2vftr.h"

int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[]) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Waitall(count, array_of_requests, array_of_statuses);
   } else {
      return vftr_MPI_Waitall_c2vftr(count, array_of_requests, array_of_statuses);
   }
}

#endif
