#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "ineighbor_allgatherv_c2vftr.h"

int MPI_Ineighbor_allgatherv(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             const int *recvcounts, const int *displs,
                             MPI_Datatype recvtype, MPI_Comm comm,
                             MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ineighbor_allgatherv(sendbuf, sendcount, sendtype,
                                       recvbuf, recvcounts, displs,
                                       recvtype, comm, request);
   } else {
      return vftr_MPI_Ineighbor_allgatherv_c2vftr(sendbuf, sendcount, sendtype,
                                                  recvbuf, recvcounts, displs,
                                                  recvtype, comm, request);
   }
}

#endif
