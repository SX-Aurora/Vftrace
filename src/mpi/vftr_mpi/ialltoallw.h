#ifndef IALLTOALLW_H
#define IALLTOALLW_H

#include <mpi.h>

int vftr_MPI_Ialltoallw(const void *sendbuf, const int *sendcounts,
                        const int *sdispls, const MPI_Datatype *sendtypes,
                        void *recvbuf, const int *recvcounts,
                        const int *rdispls, const MPI_Datatype *recvtypes,
                        MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ialltoallw_inplace(const void *sendbuf, const int *sendcounts,
                                const int *sdispls, const MPI_Datatype *sendtypes,
                                void *recvbuf, const int *recvcounts,
                                const int *rdispls, const MPI_Datatype *recvtypes,
                                MPI_Comm comm, MPI_Request *request);

int vftr_MPI_Ialltoallw_intercom(const void *sendbuf, const int *sendcounts,
                                 const int *sdispls, const MPI_Datatype *sendtypes,
                                 void *recvbuf, const int *recvcounts,
                                 const int *rdispls, const MPI_Datatype *recvtypes,
                                 MPI_Comm comm, MPI_Request *request);

#endif
