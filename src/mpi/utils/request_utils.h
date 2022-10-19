#ifndef REQUEST_UTILS_H
#define REQUEST_UTILS_H

#include <stdbool.h>

#include <mpi.h>

// check if a request is active
bool vftr_mpi_request_is_active(MPI_Request request);

#endif
