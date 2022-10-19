#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "start_c2vftr.h"

int MPI_Start(MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Start(request);
   } else {
      return vftr_MPI_Start_c2vftr(request);
   }
}

#endif
