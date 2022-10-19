#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "reduce_c2vftr.h"

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root,
               MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Reduce_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
   } else {
      return vftr_MPI_Reduce_c2vftr(sendbuf, recvbuf, count,
                                    datatype, op, root, comm);
   }
}

#endif
