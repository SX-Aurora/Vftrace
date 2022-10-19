#ifndef SENDRECV_C2VFTR_H
#define SENDRECV_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Sendrecv_c2vftr(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, int dest, int sendtag,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int source, int recvtag, MPI_Comm comm,
                             MPI_Status *status);

#endif
#endif
