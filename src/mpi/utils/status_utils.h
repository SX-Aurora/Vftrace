#ifndef STATUS_UTILS_H
#define STATUS_UTILS_H

#include <stdbool.h>

#include <mpi.h>

// mark a MPI_Status as empty
void vftr_empty_mpi_status(MPI_Status *status);

// check if a status is empty
bool vftr_mpi_status_is_empty(MPI_Status *status);

#endif
