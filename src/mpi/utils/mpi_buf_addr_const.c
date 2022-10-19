#include <stdbool.h>

#include <mpi.h>

// check if the given address is the special MPI_BOTTOM handle
bool vftr_is_C_MPI_BOTTOM(const void *addr) {
   return addr == MPI_BOTTOM;
}

// check if the given address is the special MPI_IN_PLACE handle
bool vftr_is_C_MPI_IN_PLACE(const void *addr) {
   return addr == MPI_IN_PLACE;
}
