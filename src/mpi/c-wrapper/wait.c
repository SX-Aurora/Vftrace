#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "wait_c2vftr.h"

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Wait(request, status);
   } else {
      return vftr_MPI_Wait_c2vftr(request, status);
   }
}

#endif
