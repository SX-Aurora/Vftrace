#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_logging.h"
#include "ialltoallv_c2vftr.h"

int MPI_Ialltoallv(const void *sendbuf, const int *sendcounts,
                   const int *sdispls, MPI_Datatype sendtype,
                   void *recvbuf, const int *recvcounts,
                   const int *rdispls, MPI_Datatype recvtype,
                   MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ialltoallv(sendbuf, sendcounts,
                             sdispls, sendtype,
                             recvbuf, recvcounts,
                             rdispls, recvtype,
                             comm, request);
   } else {
      return vftr_MPI_Ialltoallv_c2vftr(sendbuf, sendcounts,
                                        sdispls, sendtype,
                                        recvbuf, recvcounts,
                                        rdispls, recvtype,
                                        comm, request);
   }
}

#endif
