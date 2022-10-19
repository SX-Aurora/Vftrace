#ifndef INEIGHBOR_ALLGATHERV_C2VFTR_H
#define INEIGHBOR_ALLGATHERV_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Ineighbor_allgatherv_c2vftr(const void *sendbuf, int sendcount,
                                         MPI_Datatype sendtype, void *recvbuf,
                                         const int *recvcounts, const int *displs,
                                         MPI_Datatype recvtype, MPI_Comm comm,
                                         MPI_Request *request);

#endif
#endif
