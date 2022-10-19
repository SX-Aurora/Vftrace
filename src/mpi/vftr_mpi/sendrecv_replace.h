#ifndef SENDRECV_REPLACE_H
#define SENDRECV_REPLACE_H

#include <mpi.h>

int vftr_MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
                              int dest, int sendtag, int source, int recvtag,
                              MPI_Comm comm, MPI_Status *status);

#endif
