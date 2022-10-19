#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "isend_c2vftr.h"

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm,
              MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
   } else {
      return vftr_MPI_Isend_c2vftr(buf, count, datatype, dest, tag, comm, request);
   }
}

#endif
