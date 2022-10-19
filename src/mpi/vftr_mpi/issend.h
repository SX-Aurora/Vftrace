#ifndef ISSEND_H
#define ISSEND_H

#include <mpi.h>

int vftr_MPI_Issend(const void *buf, int count, MPI_Datatype datatype,
                    int dest, int tag, MPI_Comm comm,
                    MPI_Request *request);

#endif
