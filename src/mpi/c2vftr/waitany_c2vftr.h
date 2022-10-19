#ifndef WAITANY_C2VFTR_H
#define WAITANY_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Waitany_c2vftr(int count, MPI_Request array_of_requests[],
                            int *index, MPI_Status *status);

#endif
#endif
