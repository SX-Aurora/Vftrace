#ifndef MPI_STATE_TYPES_H
#define MPI_STATE_TYPES_H

#include <stdbool.h>

#include "request_types.h"

typedef struct {
   // PControl level as required
   // by the MPI-Standard for profiling interfaces
   int pcontrol_level;
   // list to store open requests from non-blocking and persistent communication
   int nopen_requests;
   vftr_request_t *open_requests;
   // list of ranks to be included in the profile table
   int nprof_ranks;
   int *prof_ranks;
   bool my_rank_in_prof;
} mpi_state_t;

#endif
