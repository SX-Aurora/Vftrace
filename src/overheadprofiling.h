#ifndef OVERHEADPROFILING_H
#define OVERHEADPROFILING_H

#include "vftrace_state.h"
#include "overheadprofiling_types.h"

overheadProfile_t vftr_new_overheadprofiling();

void vftr_accumulate_hook_overheadprofiling(overheadProfile_t *prof,
                                            long long overhead_usec);

#ifdef _MPI
void vftr_accumulate_mpi_overheadprofiling(overheadProfile_t *prof,
                                           long long overhead_usec);
#endif

#ifdef _OMP
void vftr_accumulate_omp_overheadprofiling(overheadProfile_t *prof,
                                           long long overhead_usec);
#endif

void vftr_overheadprofiling_free(overheadProfile_t *overheadprof_ptr);

overheadProfile_t *vftr_get_my_overheadProfile(vftrace_t vftrace);

#endif
