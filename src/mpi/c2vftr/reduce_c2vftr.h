#ifndef REDUCE_C2VFTR_H
#define REDUCE_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Reduce_c2vftr(const void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op, int root,
                           MPI_Comm comm);

#endif
#endif
