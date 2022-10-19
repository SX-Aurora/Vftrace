#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_logging.h"
#include "igatherv_c2vftr.h"

int MPI_Igatherv(const void *sendbuf, int sendcount,
                 MPI_Datatype sendtype, void *recvbuf,
                 const int *recvcounts, const int *displs,
                 MPI_Datatype recvtype, int root,
                 MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf,
                           recvcounts, displs, recvtype, root, comm,
                           request);
   } else {
      return vftr_MPI_Igatherv_c2vftr(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcounts, displs,
                                      recvtype, root, comm, request);
   }
}

#endif
