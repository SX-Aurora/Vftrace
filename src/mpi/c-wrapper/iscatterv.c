#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_logging.h"
#include "iscatterv_c2vftr.h"

int MPI_Iscatterv(const void *sendbuf, const int *sendcounts,
                  const int *displs, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount,
                  MPI_Datatype recvtype, int root,
                  MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype,
                            recvbuf, recvcount, recvtype, root, comm,
                            request);
   } else {
      return vftr_MPI_Iscatterv_c2vftr(sendbuf, sendcounts, displs,
                                       sendtype, recvbuf, recvcount,
                                       recvtype, root, comm, request);
   }
}

#endif
