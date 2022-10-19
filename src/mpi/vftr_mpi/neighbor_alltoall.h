#ifndef NEIGHBOR_ALLTOALL_H
#define NEIGHBOR_ALLTOALL_H

#include <mpi.h>

int vftr_MPI_Neighbor_alltoall_graph(const void *sendbuf, int sendcount,
                                     MPI_Datatype sendtype, void *recvbuf,
                                     int recvcount, MPI_Datatype recvtype,
                                     MPI_Comm comm);

int vftr_MPI_Neighbor_alltoall_cart(const void *sendbuf, int sendcount,
                                    MPI_Datatype sendtype, void *recvbuf,
                                    int recvcount, MPI_Datatype recvtype,
                                    MPI_Comm comm);

int vftr_MPI_Neighbor_alltoall_dist_graph(const void *sendbuf, int sendcount,
                                          MPI_Datatype sendtype, void *recvbuf,
                                          int recvcount, MPI_Datatype recvtype,
                                          MPI_Comm comm);

#endif
