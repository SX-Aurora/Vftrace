#ifndef INEIGHBOR_ALLTOALLW_H
#define INEIGHBOR_ALLTOALLW_H

#include <mpi.h>

int vftr_MPI_Ineighbor_alltoallw_graph(const void *sendbuf, const int *sendcounts,
                                       const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                                       void *recvbuf, const int *recvcounts,
                                       const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                                       MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ineighbor_alltoallw_cart(const void *sendbuf, const int *sendcounts,
                                      const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                                      void *recvbuf, const int *recvcounts,
                                      const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                                      MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ineighbor_alltoallw_dist_graph(const void *sendbuf, const int *sendcounts,
                                            const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                                            void *recvbuf, const int *recvcounts,
                                            const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                                            MPI_Comm comm, MPI_Request *request);

#endif
