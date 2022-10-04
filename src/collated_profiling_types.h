#ifndef COLLATED_PROFILING_TYPES_H
#define COLLATED_PROFILING_TYPES_H

#include "collated_callprofiling_types.h"
#ifdef _MPI
#include "mpiprofiling_types.h"
#endif
#ifdef _CUPTI
#include "cuptiprofiling_types.h"
#endif

typedef struct {
   collated_callprofile_t callprof;
#ifdef _MPI
   mpiprofile_t mpiprof;
#endif
#ifdef _CUPTI
   cuptiprofile_t cuptiprof;
#endif
} collated_profile_t;

#endif
