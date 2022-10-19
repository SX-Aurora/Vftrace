#ifndef INEIGHBOR_ALLGATHERV_H
#define INEIGHBOR_ALLGATHERV_H

#include <mpi.h>

int vftr_MPI_Ineighbor_allgatherv_graph(const void *sendbuf, int sendcount,
                                        MPI_Datatype sendtype, void *recvbuf,
                                        const int *recvcounts, const int *displs,
                                        MPI_Datatype recvtype, MPI_Comm comm,
                                        MPI_Request *request);

int vftr_MPI_Ineighbor_allgatherv_cart(const void *sendbuf, int sendcount,
                                       MPI_Datatype sendtype, void *recvbuf,
                                       const int *recvcounts, const int *displs,
                                       MPI_Datatype recvtype, MPI_Comm comm,
                                       MPI_Request *request);

int vftr_MPI_Ineighbor_allgatherv_dist_graph(const void *sendbuf, int sendcount,
                                             MPI_Datatype sendtype, void *recvbuf,
                                             const int *recvcounts, const int *displs,
                                             MPI_Datatype recvtype, MPI_Comm comm,
                                             MPI_Request *request);

#endif
