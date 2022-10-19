#ifndef RSEND_INIT_H
#define RSEND_INIT_H

#include <mpi.h>

int vftr_MPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype,
                        int dest, int tag, MPI_Comm comm,
                        MPI_Request *request);

#endif
