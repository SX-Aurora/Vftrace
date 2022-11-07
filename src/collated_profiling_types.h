#ifndef COLLATED_PROFILING_TYPES_H
#define COLLATED_PROFILING_TYPES_H

#include "collated_callprofiling_types.h"
#ifdef _MPI
#include "mpiprofiling_types.h"
#endif
#ifdef _OMP
#include "ompprofiling_types.h"
#endif
#ifdef _CUDA
#include "cudaprofiling_types.h"
#endif
#ifdef _ACCPROF
#include "accprofiling_types.h"
#endif

typedef struct {
   collated_callprofile_t callprof;
#ifdef _MPI
   mpiprofile_t mpiprof;
#endif
#ifdef _OMP
   ompprofile_t ompprof;
#endif
#ifdef _CUDA
   cudaprofile_t cudaprof;
#endif
#ifdef _ACCPROF
   accprofile_t accprof;
#endif
} collated_profile_t;

#endif
