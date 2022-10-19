#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "testany_c2vftr.h"

int MPI_Testany(int count, MPI_Request array_of_requests[],
                int *index, int *flag, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Testany(count, array_of_requests, index, flag, status);
   } else {
      return vftr_MPI_Testany_c2vftr(count, array_of_requests, index, flag, status);
   }
}

#endif
