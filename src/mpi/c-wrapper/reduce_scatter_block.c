#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "reduce_scatter_block_c2vftr.h"

int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf,
                             int recvcount, MPI_Datatype datatype,
                             MPI_Op op, MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Reduce_scatter_block_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
   } else {
      return vftr_MPI_Reduce_scatter_block_c2vftr(sendbuf, recvbuf,
                                                  recvcount, datatype,
                                                  op, comm);
   }
}

#endif
