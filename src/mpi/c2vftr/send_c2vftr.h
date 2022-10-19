#ifndef SEND_C2VFTR_H
#define SEND_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Send_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                         int dest, int tag, MPI_Comm comm);

#endif
#endif
