#ifndef MPIPROFILING_H
#define MPIPROFILING_H

#include <stdio.h>

#include "mpi_util_types.h"
#include "mpiprofiling_types.h"
#ifdef _MPI
#include "mpi_state_types.h"
#endif
#include "process_types.h"
#include "environment_types.h"

mpiprofile_t vftr_new_mpiprofiling();

void vftr_mpiprofiling_free(mpiprofile_t *prof_ptr);

#ifdef _MPI
void vftr_accumulate_message_info(mpiprofile_t *prof_ptr,
                                  mpi_state_t mpi_state,
                                  message_direction dir,
                                  long long count,
                                  int type_idx, int type_size,
                                  int rank, int tag,
                                  long long tstart,
                                  long long tend);

void vftr_accumulate_mpiprofiling_overhead(mpiprofile_t *prof,
                                           long long overhead_nsec);

mpiprofile_t vftr_add_mpiprofiles(mpiprofile_t profA, mpiprofile_t profB);

void vftr_create_profiled_ranks_list(environment_t environment,
                                     process_t process,
                                     mpi_state_t *mpi_state);

void vftr_free_profiled_ranks_list(mpi_state_t *mpi_state);

void vftr_print_mpiprofiling(FILE *fp, mpiprofile_t mpiprof);

#endif

long long *vftr_get_total_mpi_overhead(stacktree_t stacktree, int nthreads);

long long vftr_get_total_collated_mpi_overhead(collated_stacktree_t stacktree);

#endif
