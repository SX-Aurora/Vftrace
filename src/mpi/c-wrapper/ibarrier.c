#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "ibarrier_c2vftr.h"

int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ibarrier(comm, request);
   } else {
      return vftr_MPI_Ibarrier_c2vftr(comm, request);
   }
}

#endif
