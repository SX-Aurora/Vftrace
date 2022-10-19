#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "issend_c2vftr.h"

int MPI_Issend(const void *buf, int count, MPI_Datatype datatype,
               int dest, int tag, MPI_Comm comm,
               MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Issend(buf, count, datatype, dest, tag, comm, request);
   } else {
      return vftr_MPI_Issend_c2vftr(buf, count, datatype, dest, tag, comm, request);
   }
}

#endif
