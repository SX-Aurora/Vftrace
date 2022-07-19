#ifndef MPIPROFILING_H
#define MPIPROFILING_H

#include "mpi_util_types.h"
#include "mpiprofiling_types.h"
#ifdef _MPI
#include "mpi_state_types.h"
#endif

mpiProfile_t vftr_new_mpiprofiling();

void vftr_accumulate_message_info(mpiProfile_t *prof_ptr,
                                  mpi_state_t mpi_state,
                                  message_direction dir,
                                  long long count,
                                  int type_idx, int type_size,
                                  int rank, int tag,
                                  long long tstart,
                                  long long tend);

void vftr_mpiprofiling_free(mpiProfile_t *prof_ptr);

#ifdef _MPI
void vftr_create_profiled_ranks_list(environment_t environment,
                                     process_t process,
                                     mpi_state_t *mpi_state);

void vftr_free_profiled_ranks_list(mpi_state_t *mpi_state);
#endif

#endif
