#ifndef ALLTOALLW_H
#define ALLTOALLW_H

#include <mpi.h>

int vftr_MPI_Alltoallw(const void *sendbuf, const int *sendcounts,
                       const int *sdispls, const MPI_Datatype *sendtypes,
                       void *recvbuf, const int *recvcounts,
                       const int *rdispls, const MPI_Datatype *recvtypes,
                       MPI_Comm comm);

int vftr_MPI_Alltoallw_inplace(const void *sendbuf, const int *sendcounts,
                               const int *sdispls, const MPI_Datatype *sendtypes,
                               void *recvbuf, const int *recvcounts,
                               const int *rdispls, const MPI_Datatype *recvtypes,
                               MPI_Comm comm);

int vftr_MPI_Alltoallw_intercom(const void *sendbuf, const int *sendcounts,
                                const int *sdispls, const MPI_Datatype *sendtypes,
                                void *recvbuf, const int *recvcounts,
                                const int *rdispls, const MPI_Datatype *recvtypes,
                                MPI_Comm comm);

#endif
