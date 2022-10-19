#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sync_time.h"
#include "alltoall_c2vftr.h"

int MPI_Alltoall(const void *sendbuf, int sendcount,
                 MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {

   vftr_estimate_sync_time("MPI_Alltoall_sync", comm);

   if (vftr_no_mpi_logging()) {
      return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                           recvcount, recvtype, comm);
   } else {
      return vftr_MPI_Alltoall_c2vftr(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcount, recvtype,
                                      comm);
   }
}

#endif
