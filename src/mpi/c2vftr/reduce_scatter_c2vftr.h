#ifndef REDUCE_SCATTER_C2VFTR_H
#define REDUCE_SCATTER_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Reduce_scatter_c2vftr(const void *sendbuf, void *recvbuf,
                                   const int *recvcounts, MPI_Datatype datatype,
                                   MPI_Op op, MPI_Comm comm);

#endif
#endif
