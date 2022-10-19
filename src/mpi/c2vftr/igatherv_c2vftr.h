#ifndef IGATHERV_C2VFTR_H
#define IGATHERV_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Igatherv_c2vftr(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             const int *recvcounts, const int *displs,
                             MPI_Datatype recvtype, int root,
                             MPI_Comm comm, MPI_Request *request);

#endif
#endif
