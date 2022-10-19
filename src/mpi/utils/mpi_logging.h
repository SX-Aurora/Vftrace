#ifndef MPI_LOGGING_H
#define MPI_LOGGING_H

#include <stdbool.h>

#include "mpi_util_types.h"

// determine based on several criteria if
// the communication should just be executed or also logged
bool vftr_no_mpi_logging();

// int version of above function for well defined fortran-interoperability
int vftr_no_mpi_logging_int();

// write the message information to the vfd-file
void vftr_write_message_info(message_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend,
                             int stackID, int threadID);

#endif
