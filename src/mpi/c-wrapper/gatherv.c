#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "gatherv_c2vftr.h"

int MPI_Gatherv(const void *sendbuf, int sendcount,
                MPI_Datatype sendtype, void *recvbuf,
                const int *recvcounts, const int *displs,
                MPI_Datatype recvtype, int root,
                MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Gatherv_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf,
                          recvcounts, displs, recvtype, root, comm);
   } else {
      return vftr_MPI_Gatherv_c2vftr(sendbuf, sendcount, sendtype,
                                     recvbuf, recvcounts, displs,
                                     recvtype, root, comm);
   }
}

#endif
