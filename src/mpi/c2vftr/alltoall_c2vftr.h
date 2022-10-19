#ifndef ALLTOALL_C2VFTR_H
#define ALLTOALL_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Alltoall_c2vftr(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             MPI_Comm comm);

#endif
#endif
