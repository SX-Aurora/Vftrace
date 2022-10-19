#ifndef SENDRECV_H
#define SENDRECV_H

#include <mpi.h>

int vftr_MPI_Sendrecv(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, int dest, int sendtag,
                      void *recvbuf, int recvcount, MPI_Datatype recvtype,
                      int source, int recvtag, MPI_Comm comm,
                      MPI_Status *status);

#endif
