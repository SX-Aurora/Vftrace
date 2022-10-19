#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "ibcast_c2vftr.h"

int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype,
               int root, MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ibcast(buffer, count, datatype, root, comm, request);
   } else {
      return vftr_MPI_Ibcast_c2vftr(buffer, count, datatype, root, comm, request);
   }
}

#endif
