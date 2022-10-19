#ifndef RECV_INIT_C2VFTR_H
#define RECV_INIT_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Recv_init_c2vftr(void *buf, int count, MPI_Datatype datatype,
                              int source, int tag, MPI_Comm comm,
                              MPI_Request *request);

#endif
#endif
