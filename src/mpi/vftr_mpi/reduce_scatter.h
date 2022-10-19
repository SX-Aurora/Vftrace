#ifndef REDUCE_SCATTER_H
#define REDUCE_SCATTER_H

#include <mpi.h>

int vftr_MPI_Reduce_scatter(const void *sendbuf, void *recvbuf,
                            const int *recvcounts, MPI_Datatype datatype,
                            MPI_Op op, MPI_Comm comm);

int vftr_MPI_Reduce_scatter_inplace(const void *sendbuf, void *recvbuf,
                                    const int *recvcounts, MPI_Datatype datatype,
                                    MPI_Op op, MPI_Comm comm);

int vftr_MPI_Reduce_scatter_intercom(const void *sendbuf, void *recvbuf,
                                     const int *recvcounts, MPI_Datatype datatype,
                                     MPI_Op op, MPI_Comm comm);

#endif
