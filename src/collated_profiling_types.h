#ifndef COLLATED_PROFILING_TYPES_H
#define COLLATED_PROFILING_TYPES_H

#include "collated_callprofiling_types.h"
#ifdef _MPI
#include "mpiprofiling_types.h"
#endif
#ifdef _OMP
#include "ompprofiling_types.h"
#endif
#ifdef _CUPTI
#include "cuptiprofiling_types.h"
#endif
#ifdef _VEDA
#include "vedaprofiling_types.h"
#endif

typedef struct {
   collated_callprofile_t callprof;
#ifdef _MPI
   mpiprofile_t mpiprof;
#endif
#ifdef _OMP
   ompprofile_t ompprof;
#endif
#ifdef _CUPTI
   cuptiprofile_t cuptiprof;
#endif
#ifdef _VEDA
   vedaprofile_t vedaprof;
#endif
} collated_profile_t;

#endif
