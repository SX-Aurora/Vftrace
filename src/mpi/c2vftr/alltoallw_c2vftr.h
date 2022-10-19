#ifndef ALLTOALLW_C2VFTR_H
#define ALLTOALLW_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Alltoallw_c2vftr(const void *sendbuf, const int *sendcounts,
                              const int *sdispls, const MPI_Datatype *sendtypes,
                              void *recvbuf, const int *recvcounts,
                              const int *rdispls, const MPI_Datatype *recvtypes,
                              MPI_Comm comm);

#endif
#endif
