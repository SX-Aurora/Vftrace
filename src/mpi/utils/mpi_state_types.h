#ifndef MPI_STATE_TYPES_H
#define MPI_STATE_TYPES_H

typedef struct {
   // PControl level as required
   // by the MPI-Standard for profiling interfaces
   int pcontrol_level;
} mpi_state_t;

#endif
