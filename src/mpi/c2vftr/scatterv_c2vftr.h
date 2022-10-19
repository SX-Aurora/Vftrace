#ifndef SCATTERV_C2VFTR_H
#define SCATTERV_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Scatterv_c2vftr(const void *sendbuf, const int *sendcounts,
                             const int *displs, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount,
                             MPI_Datatype recvtype,
                             int root, MPI_Comm comm);

#endif
#endif
