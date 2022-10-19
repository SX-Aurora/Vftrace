#ifndef IREDUCE_SCATTER_BLOCK_C2VFTR_H
#define IREDUCE_SCATTER_BLOCK_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Ireduce_scatter_block_c2vftr(const void *sendbuf, void *recvbuf,
                                          int recvcount, MPI_Datatype datatype,
                                          MPI_Op op, MPI_Comm comm,
                                          MPI_Request *request);

#endif
#endif
