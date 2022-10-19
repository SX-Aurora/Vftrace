#ifndef BSEND_INIT_H
#define BSEND_INIT_H

#include <mpi.h>

int vftr_MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype,
                        int dest, int tag, MPI_Comm comm,
                        MPI_Request *request);

#endif
