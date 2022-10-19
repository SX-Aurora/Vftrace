#ifndef IREDUCE_SCATTER_H
#define IREDUCE_SCATTER_H

#include <mpi.h>

int vftr_MPI_Ireduce_scatter(const void *sendbuf, void *recvbuf,
                             const int *recvcounts, MPI_Datatype datatype,
                             MPI_Op op, MPI_Comm comm,
                             MPI_Request *request);

int vftr_MPI_Ireduce_scatter_inplace(const void *sendbuf, void *recvbuf,
                                     const int *recvcounts, MPI_Datatype datatype,
                                     MPI_Op op, MPI_Comm comm,
                                     MPI_Request *request);

int vftr_MPI_Ireduce_scatter_intercom(const void *sendbuf, void *recvbuf,
                                      const int *recvcounts, MPI_Datatype datatype,
                                      MPI_Op op, MPI_Comm comm,
                                      MPI_Request *request);

#endif
