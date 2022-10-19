#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "mpi_logging.h"
#include "ineighbor_alltoallw_c2vftr.h"

int MPI_Ineighbor_alltoallw(const void *sendbuf, const int *sendcounts,
                            const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                            void *recvbuf, const int *recvcounts,
                            const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                            MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ineighbor_alltoallw(sendbuf, sendcounts,
                                      sdispls, sendtypes,
                                      recvbuf, recvcounts,
                                      rdispls, recvtypes,
                                      comm, request);
   } else {
      return vftr_MPI_Ineighbor_alltoallw_c2vftr(sendbuf, sendcounts,
                                                 sdispls, sendtypes,
                                                 recvbuf, recvcounts,
                                                 rdispls, recvtypes,
                                                 comm, request);
   }
}

#endif
