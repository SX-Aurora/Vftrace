#ifndef OVERHEADPROFILING_H
#define OVERHEADPROFILING_H

#include "vftrace_state.h"
#include "overheadprofiling_types.h"

overheadProfile_t vftr_new_overheadprofiling();

void vftr_accumulate_hook_overheadprofiling(overheadProfile_t *prof,
                                            long long overhead_usec);

long long *vftr_get_total_hook_overhead(stacktree_t stacktree, int nthreads);

#ifdef _MPI
void vftr_accumulate_mpi_overheadprofiling(overheadProfile_t *prof,
                                           long long overhead_usec);

long long *vftr_get_total_mpi_overhead(stacktree_t stacktree, int nthreads);
#endif

#ifdef _OMP
void vftr_accumulate_omp_overheadprofiling(overheadProfile_t *prof,
                                           long long overhead_usec);

long long *vftr_get_total_omp_overhead(stacktree_t stacktree, int nthreads);
#endif

void vftr_overheadprofiling_free(overheadProfile_t *overheadprof_ptr);

overheadProfile_t *vftr_get_my_overheadProfile(vftrace_t vftrace);

#endif
