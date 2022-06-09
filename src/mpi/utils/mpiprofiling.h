#ifndef MPIPROFILING_H
#define MPIPROFILING_H

#include "mpi_util_types.h"
#include "mpiprofiling_types.h"

mpiProfile_t vftr_new_mpiprofiling();

void vftr_accumulate_message_info(mpiProfile_t *prof_ptr,
                                  message_direction dir, long long count,
                                  int type_idx, int type_size, int rank,
                                  int tag, long long tstart,
                                  long long tend);

void vftr_mpiprofiling_free(mpiProfile_t *prof_ptr);

#endif
