#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "ireduce_c2vftr.h"

int MPI_Ireduce(const void *sendbuf, void *recvbuf, int count,
                MPI_Datatype datatype, MPI_Op op, int root,
                MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ireduce(sendbuf, recvbuf, count, datatype,
                          op, root, comm, request);
   } else {
      return vftr_MPI_Ireduce_c2vftr(sendbuf, recvbuf, count, datatype,
                                     op, root, comm, request);
   }
}

#endif
