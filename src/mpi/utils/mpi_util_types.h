#ifndef VFTR_MPI_UTIL_TYPES_H
#define VFTR_MPI_UTIL_TYPES_H

//#include <stdbool.h>

#ifdef _MPI
#include <mpi.h>
#endif

typedef enum {
   send,
   recv
} message_direction;

struct vftr_mpi_type_t {
#ifdef _MPI
   MPI_Datatype mpi_type;
#else
   int mpi_type;
#endif
   char *name;
};

#ifdef _MPI
// Translates an MPI-Datatype into the vftr type index
int vftr_get_mpitype_idx(MPI_Datatype mpi_type);

// Converts an mpi-datatype into a name string for that type
const char *vftr_get_mpitype_string(MPI_Datatype mpi_type);
#endif

// Converts an mpi-datatype into a name string for that type
const char *vftr_get_mpitype_string_from_idx(int mpi_type_idx);

#endif
