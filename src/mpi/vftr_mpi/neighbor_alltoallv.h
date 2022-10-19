#ifndef NEIGHBOR_ALLTOALLV_H
#define NEIGHBOR_ALLTOALLV_H

#include <mpi.h>

int vftr_MPI_Neighbor_alltoallv_graph(const void *sendbuf, const int *sendcounts,
                                      const int *sdispls, MPI_Datatype sendtype,
                                      void *recvbuf, const int *recvcounts,
                                      const int *rdispls, MPI_Datatype recvtype,
                                      MPI_Comm comm);

int vftr_MPI_Neighbor_alltoallv_cart(const void *sendbuf, const int *sendcounts,
                                     const int *sdispls, MPI_Datatype sendtype,
                                     void *recvbuf, const int *recvcounts,
                                     const int *rdispls, MPI_Datatype recvtype,
                                     MPI_Comm comm);

int vftr_MPI_Neighbor_alltoallv_dist_graph(const void *sendbuf, const int *sendcounts,
                                           const int *sdispls, MPI_Datatype sendtype,
                                           void *recvbuf, const int *recvcounts,
                                           const int *rdispls, MPI_Datatype recvtype,
                                           MPI_Comm comm);

#endif
