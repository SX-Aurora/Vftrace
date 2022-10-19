#ifndef INEIGHBOR_ALLTOALL_H
#define INEIGHBOR_ALLTOALL_H

#include <mpi.h>

int vftr_MPI_Ineighbor_alltoall_graph(const void *sendbuf, int sendcount,
                                      MPI_Datatype sendtype, void *recvbuf,
                                      int recvcount, MPI_Datatype recvtype,
                                      MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ineighbor_alltoall_cart(const void *sendbuf, int sendcount,
                                     MPI_Datatype sendtype, void *recvbuf,
                                     int recvcount, MPI_Datatype recvtype,
                                     MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ineighbor_alltoall_dist_graph(const void *sendbuf, int sendcount,
                                           MPI_Datatype sendtype, void *recvbuf,
                                           int recvcount, MPI_Datatype recvtype,
                                           MPI_Comm comm, MPI_Request *request);

#endif
