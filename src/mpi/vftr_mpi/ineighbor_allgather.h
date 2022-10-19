#ifndef INEIGHBOR_ALLGATHER_H
#define INEIGHBOR_ALLGATHER_H

#include <mpi.h>

int vftr_MPI_Ineighbor_allgather_graph(const void *sendbuf, int sendcount,
                                       MPI_Datatype sendtype, void *recvbuf,
                                       int recvcount, MPI_Datatype recvtype,
                                       MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ineighbor_allgather_cart(const void *sendbuf, int sendcount,
                                      MPI_Datatype sendtype, void *recvbuf,
                                      int recvcount, MPI_Datatype recvtype,
                                      MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ineighbor_allgather_dist_graph(const void *sendbuf, int sendcount,
                                            MPI_Datatype sendtype, void *recvbuf,
                                            int recvcount, MPI_Datatype recvtype,
                                            MPI_Comm comm, MPI_Request *request);

#endif
