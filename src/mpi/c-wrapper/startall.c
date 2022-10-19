#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "startall_c2vftr.h"

int MPI_Startall(int count, MPI_Request *array_of_requests) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Startall(count, array_of_requests);
   } else {
      return vftr_MPI_Startall_c2vftr(count, array_of_requests);
   }
}

#endif
