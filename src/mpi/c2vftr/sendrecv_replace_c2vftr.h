#ifndef SENDRECV_REPLACE_C2VFTR_H
#define SENDRECV_REPLACE_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Sendrecv_replace_c2vftr(void *buf, int count, MPI_Datatype datatype,
                                     int dest, int sendtag, int source, int recvtag,
                                     MPI_Comm comm, MPI_Status *status);

#endif
#endif
