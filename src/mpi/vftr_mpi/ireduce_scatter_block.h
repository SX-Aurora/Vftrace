#ifndef IREDUCE_SCATTER_BLOCK_H
#define IREDUCE_SCATTER_BLOCK_H

#include <mpi.h>

int vftr_MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf,
                                   int recvcount, MPI_Datatype datatype,
                                   MPI_Op op, MPI_Comm comm,
                                   MPI_Request *request);

int vftr_MPI_Ireduce_scatter_block_inplace(const void *sendbuf, void *recvbuf,
                                           int recvcount, MPI_Datatype datatype,
                                           MPI_Op op, MPI_Comm comm,
                                           MPI_Request *request);

int vftr_MPI_Ireduce_scatter_block_intercom(const void *sendbuf, void *recvbuf,
                                            int recvcount, MPI_Datatype datatype,
                                            MPI_Op op, MPI_Comm comm,
                                            MPI_Request *request);

#endif
