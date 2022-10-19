#ifdef _MPI

#include <stdlib.h>

#include <mpi.h>

#include "mpi_logging.h"
#include "ireduce_scatter_c2vftr.h"

int MPI_Ireduce_scatter(const void *sendbuf, void *recvbuf,
                        const int *recvcounts, MPI_Datatype datatype,
                        MPI_Op op, MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts,
                                  datatype, op, comm, request);
   } else {
      return vftr_MPI_Ireduce_scatter_c2vftr(sendbuf, recvbuf, recvcounts,
                                             datatype, op, comm, request);
   }
}

#endif
