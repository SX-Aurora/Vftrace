#ifndef NEIGHBOR_ALLTOALLW_C2VFTR_H
#define NEIGHBOR_ALLTOALLW_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Neighbor_alltoallw_c2vftr(const void *sendbuf, const int *sendcounts,
                                       const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                                       void *recvbuf, const int *recvcounts,
                                       const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                                       MPI_Comm comm);

#endif
#endif
