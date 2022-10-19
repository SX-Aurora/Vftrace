#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "iexscan_c2vftr.h"

int MPI_Iexscan(const void *sendbuf, void *recvbuf, int count,
                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Iexscan(sendbuf, recvbuf, count, datatype, op, comm, request);
   } else {
      return vftr_MPI_Iexscan_c2vftr(sendbuf, recvbuf, count, datatype, op, comm, request);
   }
}

#endif
