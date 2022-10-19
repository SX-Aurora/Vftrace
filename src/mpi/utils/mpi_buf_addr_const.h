#include <stdbool.h>

#ifndef VFTR_MPI_BUF_ADDR_CONST_H
#define VFTR_MPI_BUF_ADDR_CONST_H

#include <mpi.h>

// check if the given address is the special MPI_BOTTOM handle
bool vftr_is_C_MPI_BOTTOM(const void *addr);

bool vftr_is_F_MPI_BOTTOM(const void *addr);

// check if the given address is the special MPI_IN_PLACE handle
bool vftr_is_C_MPI_IN_PLACE(const void *addr);

bool vftr_is_F_MPI_IN_PLACE(const void *addr);

#endif
