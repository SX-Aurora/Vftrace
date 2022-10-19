#ifndef WAITALL_C2VFTR_H
#define WAITALL_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Waitall_c2vftr(int count, MPI_Request array_of_requests[],
                            MPI_Status array_of_statuses[]);

#endif
#endif
