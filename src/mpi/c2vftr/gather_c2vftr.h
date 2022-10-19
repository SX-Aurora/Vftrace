#ifndef GATHER_C2VFTR_H
#define GATHER_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Gather_c2vftr(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf,
                           int recvcount, MPI_Datatype recvtype,
                           int root, MPI_Comm comm);

#endif
#endif
