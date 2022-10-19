#ifndef IREDUCE_SCATTER_C2VFTR_H
#define IREDUCE_SCATTER_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Ireduce_scatter_c2vftr(const void *sendbuf, void *recvbuf,
                                    const int *recvcounts, MPI_Datatype datatype,
                                    MPI_Op op, MPI_Comm comm, MPI_Request *request);

#endif
#endif
