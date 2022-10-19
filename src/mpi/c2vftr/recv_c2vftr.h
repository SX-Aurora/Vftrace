#ifndef RECV_C2VFTR_H
#define RECV_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Recv_c2vftr(void *buf, int count, MPI_Datatype datatype,
                         int source, int tag, MPI_Comm comm, MPI_Status *status);

#endif
#endif
