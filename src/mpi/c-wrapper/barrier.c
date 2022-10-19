#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "barrier_c2vftr.h"

int MPI_Barrier(MPI_Comm comm) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Barrier(comm);
   } else {
      return vftr_MPI_Barrier_c2vftr(comm);
   }
}

#endif
