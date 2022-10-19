#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "alltoallw_c2vftr.h"

int MPI_Alltoallw(const void *sendbuf, const int *sendcounts,
                  const int *sdispls, const MPI_Datatype *sendtypes,
                  void *recvbuf, const int *recvcounts,
                  const int *rdispls, const MPI_Datatype *recvtypes,
                  MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Alltoallw_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Alltoallw(sendbuf, sendcounts,
                            sdispls, sendtypes,
                            recvbuf, recvcounts,
                            rdispls, recvtypes,
                            comm);
   } else {
      return vftr_MPI_Alltoallw_c2vftr(sendbuf, sendcounts,
                                       sdispls, sendtypes,
                                       recvbuf, recvcounts,
                                       rdispls, recvtypes,
                                       comm);
   }
}

#endif
