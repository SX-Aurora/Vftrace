#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "neighbor_allgatherv_c2vftr.h"

int MPI_Neighbor_allgatherv(const void *sendbuf, int sendcount,
                            MPI_Datatype sendtype, void *recvbuf,
                            const int *recvcounts, const int *displs,
                            MPI_Datatype recvtype, MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Neighbor_allgatherv_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Neighbor_allgatherv(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcounts, displs, recvtype,
                                      comm);
   } else {
      return vftr_MPI_Neighbor_allgatherv_c2vftr(sendbuf, sendcount, sendtype,
                                                 recvbuf, recvcounts, displs, recvtype,
                                                 comm);
   }
}

#endif
