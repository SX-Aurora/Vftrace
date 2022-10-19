#ifndef REDUCE_SCATTER_BLOCK_H
#define REDUCE_SCATTER_BLOCK_H

#include <mpi.h>

int vftr_MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf,
                                  int recvcount, MPI_Datatype datatype,
                                  MPI_Op op, MPI_Comm comm);

int vftr_MPI_Reduce_scatter_block_inplace(const void *sendbuf, void *recvbuf,
                                          int recvcount, MPI_Datatype datatype,
                                          MPI_Op op, MPI_Comm comm);

int vftr_MPI_Reduce_scatter_block_intercom(const void *sendbuf, void *recvbuf,
                                           int recvcount, MPI_Datatype datatype,
                                           MPI_Op op, MPI_Comm comm);

#endif
