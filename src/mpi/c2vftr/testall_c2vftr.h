#ifndef TESTALL_C2VFTR_H
#define TESTALL_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Testall_c2vftr(int count, MPI_Request array_of_requests[],
                            int *flag, MPI_Status array_of_statuses[]);

#endif
#endif
