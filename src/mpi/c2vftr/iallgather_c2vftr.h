#ifndef IALLGATHER_C2VFTR_H
#define IALLGATHER_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Iallgather_c2vftr(const void *sendbuf, int sendcount,
                               MPI_Datatype sendtype, void *recvbuf,
                               int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm, MPI_Request *request);

#endif
#endif
