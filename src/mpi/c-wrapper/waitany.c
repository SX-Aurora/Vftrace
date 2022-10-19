#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "waitany_c2vftr.h"

int MPI_Waitany(int count, MPI_Request array_of_requests[],
                int *index, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Waitany(count, array_of_requests, index, status);
   } else {
      return vftr_MPI_Waitany_c2vftr(count, array_of_requests, index, status);
   }
}

#endif
