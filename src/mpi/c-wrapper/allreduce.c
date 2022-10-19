#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "allreduce_c2vftr.h"

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Allreduce_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   } else {
      return vftr_MPI_Allreduce_c2vftr(sendbuf, recvbuf,
                                       count, datatype, op, comm);
   }
}

#endif
