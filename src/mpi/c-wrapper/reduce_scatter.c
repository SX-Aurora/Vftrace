#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "reduce_scatter_c2vftr.h"

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf,
                       const int *recvcounts, MPI_Datatype datatype,
                       MPI_Op op, MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Reduce_scatter_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
   } else {
      return vftr_MPI_Reduce_scatter_c2vftr(sendbuf, recvbuf,
                                            recvcounts, datatype,
                                            op, comm);
   }
}

#endif
