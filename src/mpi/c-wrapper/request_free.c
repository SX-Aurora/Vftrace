#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "request_free_c2vftr.h"

int MPI_Request_free(MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Request_free(request);
   } else {
      return vftr_MPI_Request_free_c2vftr(request);
   }
}

#endif
