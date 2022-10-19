#ifndef TESTANY_C2VFTR_H
#define TESTANY_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Testany_c2vftr(int count, MPI_Request array_of_requests[],
                            int *index, int *flag, MPI_Status *status);

#endif
#endif
