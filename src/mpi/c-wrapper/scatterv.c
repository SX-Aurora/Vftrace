#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "scatterv_c2vftr.h"

int MPI_Scatterv(const void *sendbuf, const int *sendcounts,
                 const int *displs, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount,
                 MPI_Datatype recvtype,
                 int root, MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Scatterv_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype,
                           recvbuf, recvcount, recvtype, root, comm);
   } else {
      return vftr_MPI_Scatterv_c2vftr(sendbuf, sendcounts, displs,
                                      sendtype, recvbuf, recvcount,
                                      recvtype, root, comm);
   }
}

#endif
