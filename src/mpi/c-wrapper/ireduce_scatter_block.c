#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "ireduce_scatter_block_c2vftr.h"

int MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf,
                              int recvcount, MPI_Datatype datatype,
                              MPI_Op op, MPI_Comm comm,
                              MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ireduce_scatter_block(sendbuf, recvbuf, recvcount,
                                        datatype, op, comm, request);
   } else {
      return vftr_MPI_Ireduce_scatter_block_c2vftr(sendbuf, recvbuf,
                                                   recvcount, datatype,
                                                   op, comm, request);
   }
}

#endif
