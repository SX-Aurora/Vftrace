#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "testsome_c2vftr.h"

int MPI_Testsome(int incount, MPI_Request array_of_requests[],
                 int *outcount, int array_of_indices[],
                 MPI_Status array_of_statuses[]) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Testsome(incount, array_of_requests, outcount,
                           array_of_indices, array_of_statuses);
   } else {
      return vftr_MPI_Testsome_c2vftr(incount, array_of_requests, outcount,
                                      array_of_indices, array_of_statuses);
   }
}

#endif
