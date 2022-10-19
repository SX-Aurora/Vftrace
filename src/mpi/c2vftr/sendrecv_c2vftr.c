#ifdef _MPI
#include <mpi.h>

#include "sendrecv.h"

int vftr_MPI_Sendrecv_c2vftr(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, int dest, int sendtag,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int source, int recvtag, MPI_Comm comm,
                             MPI_Status *status) {
   return vftr_MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                            recvbuf, recvcount, recvtype, source, recvtag,
                            comm, status);
}

#endif
