#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "gather_c2vftr.h"

int MPI_Gather(const void *sendbuf, int sendcount,
               MPI_Datatype sendtype, void *recvbuf,
               int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Gather_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Gather(sendbuf, sendcount, sendtype,
                         recvbuf, recvcount, recvtype,
                         root, comm);
   } else {
      return vftr_MPI_Gather_c2vftr(sendbuf, sendcount, sendtype,
                                    recvbuf, recvcount, recvtype,
                                    root, comm);
   }
}

#endif
