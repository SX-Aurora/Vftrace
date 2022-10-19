#ifndef NEIGHBOR_ALLGATHERV_H
#define NEIGHBOR_ALLGATHERV_H

#include <mpi.h>

int vftr_MPI_Neighbor_allgatherv_graph(const void *sendbuf, int sendcount,
                                       MPI_Datatype sendtype, void *recvbuf,
                                       const int *recvcount, const int *displs,
                                       MPI_Datatype recvtype, MPI_Comm comm);

int vftr_MPI_Neighbor_allgatherv_cart(const void *sendbuf, int sendcount,
                                      MPI_Datatype sendtype, void *recvbuf,
                                      const int *recvcount, const int *displs,
                                      MPI_Datatype recvtype, MPI_Comm comm);

int vftr_MPI_Neighbor_allgatherv_dist_graph(const void *sendbuf, int sendcount,
                                            MPI_Datatype sendtype, void *recvbuf,
                                            const int *recvcount, const int *displs,
                                            MPI_Datatype recvtype, MPI_Comm comm);

#endif
