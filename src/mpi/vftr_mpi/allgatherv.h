#ifndef ALLGATHERV_H
#define ALLGATHERV_H

#include <mpi.h>

int vftr_MPI_Allgatherv(const void *sendbuf, int sendcount,
                        MPI_Datatype sendtype, void *recvbuf,
                        const int *recvcounts, const int *displs,
                        MPI_Datatype recvtype, MPI_Comm comm);

int vftr_MPI_Allgatherv_inplace(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                const int *recvcounts, const int *displs,
                                MPI_Datatype recvtype, MPI_Comm comm);

int vftr_MPI_Allgatherv_intercom(const void *sendbuf, int sendcount,
                                 MPI_Datatype sendtype, void *recvbuf,
                                 const int *recvcounts, const int *displs,
                                 MPI_Datatype recvtype, MPI_Comm comm);

#endif
