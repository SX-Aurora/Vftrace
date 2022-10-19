#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_logging.h"
#include "ialltoallw_c2vftr.h"

int MPI_Ialltoallw(const void *sendbuf, const int *sendcounts,
                   const int *sdispls, const MPI_Datatype *sendtypes,
                   void *recvbuf, const int *recvcounts,
                   const int *rdispls, const MPI_Datatype *recvtypes,
                   MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ialltoallw(sendbuf, sendcounts,
                             sdispls, sendtypes,
                             recvbuf, recvcounts,
                             rdispls, recvtypes,
                             comm, request);
   } else {
      return vftr_MPI_Ialltoallw_c2vftr(sendbuf, sendcounts,
                                        sdispls, sendtypes,
                                        recvbuf, recvcounts,
                                        rdispls, recvtypes,
                                        comm, request);
   }
}

#endif
