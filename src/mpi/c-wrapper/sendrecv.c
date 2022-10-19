#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "sendrecv_c2vftr.h"

int MPI_Sendrecv(const void *sendbuf, int sendcount,
                 MPI_Datatype sendtype, int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag, MPI_Comm comm,
                 MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                           recvbuf, recvcount, recvtype, source, recvtag,
                           comm, status);
   } else {
      return vftr_MPI_Sendrecv_c2vftr(sendbuf, sendcount, sendtype, dest, sendtag,
                                      recvbuf, recvcount, recvtype, source, recvtag,
                                      comm, status);
   }
}

#endif
