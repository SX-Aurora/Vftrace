#ifndef NEIGHBOR_ALLGATHER_H
#define NEIGHBOR_ALLGATHER_H

#include <mpi.h>

int vftr_MPI_Neighbor_allgather_graph(const void *sendbuf, int sendcount,
                                      MPI_Datatype sendtype, void *recvbuf,
                                      int recvcount, MPI_Datatype recvtype,
                                      MPI_Comm comm);

int vftr_MPI_Neighbor_allgather_cart(const void *sendbuf, int sendcount,
                                     MPI_Datatype sendtype, void *recvbuf,
                                     int recvcount, MPI_Datatype recvtype,
                                     MPI_Comm comm);

int vftr_MPI_Neighbor_allgather_dist_graph(const void *sendbuf, int sendcount,
                                           MPI_Datatype sendtype, void *recvbuf,
                                           int recvcount, MPI_Datatype recvtype,
                                           MPI_Comm comm);

#endif
