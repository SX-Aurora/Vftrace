#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "neighbor_alltoallw_c2vftr.h"

int MPI_Neighbor_alltoallw(const void *sendbuf, const int *sendcounts,
                           const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                           void *recvbuf, const int *recvcounts,
                           const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                           MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Neighbor_alltoallw_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                     recvbuf, recvcounts, rdispls, recvtypes,
                                     comm);
   } else {
      return vftr_MPI_Neighbor_alltoallw_c2vftr(sendbuf, sendcounts, sdispls, sendtypes,
                                                recvbuf, recvcounts, rdispls, recvtypes,
                                                comm);
   }
}

#endif
