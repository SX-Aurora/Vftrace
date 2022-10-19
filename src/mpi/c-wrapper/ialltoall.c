#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "ialltoall_c2vftr.h"

int MPI_Ialltoall(const void *sendbuf, int sendcount,
                  MPI_Datatype sendtype, void *recvbuf,
                  int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm, MPI_Request *request) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf,
                            recvcount, recvtype, comm, request);
   } else {
      return vftr_MPI_Ialltoall_c2vftr(sendbuf, sendcount, sendtype,
                                       recvbuf, recvcount, recvtype,
                                       comm, request);
   }
}

#endif
