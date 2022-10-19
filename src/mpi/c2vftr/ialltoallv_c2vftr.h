#ifndef IALLTOALLV_C2VFTR_H
#define IALLTOALLV_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Ialltoallv_c2vftr(const void *sendbuf, const int *sendcounts,
                               const int *sdispls, MPI_Datatype sendtype,
                               void *recvbuf, const int *recvcounts,
                               const int *rdispls, MPI_Datatype recvtype,
                               MPI_Comm comm, MPI_Request *request);

#endif
#endif
