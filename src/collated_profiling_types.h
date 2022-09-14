#ifndef COLLATED_PROFILING_TYPES_H
#define COLLATED_PROFILING_TYPES_H

#include "callprofiling_types.h"
#ifdef _MPI
#include "mpiprofiling_types.h"
#endif

typedef struct {
   callProfile_t callProf;
#ifdef _MPI
   mpiProfile_t mpiProf;
#endif
} collated_profile_t;

#endif
