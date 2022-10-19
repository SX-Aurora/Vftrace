#ifndef SCATTERV_H
#define SCATTERV_H

#include <mpi.h>

int vftr_MPI_Scatterv(const void *sendbuf, const int *sendcounts,
                      const int *displs, MPI_Datatype sendtype,
                      void *recvbuf, int recvcount,
                      MPI_Datatype recvtype,
                      int root, MPI_Comm comm);

int vftr_MPI_Scatterv_inplace(const void *sendbuf, const int *sendcounts,
                              const int *displs, MPI_Datatype sendtype,
                              void *recvbuf, int recvcount,
                              MPI_Datatype recvtype,
                              int root, MPI_Comm comm);

int vftr_MPI_Scatterv_intercom(const void *sendbuf, const int *sendcounts,
                               const int *displs, MPI_Datatype sendtype,
                               void *recvbuf, int recvcount,
                               MPI_Datatype recvtype,
                               int root, MPI_Comm comm);

#endif
