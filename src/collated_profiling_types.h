#ifndef COLLATED_PROFILING_TYPES_H
#define COLLATED_PROFILING_TYPES_H

#include "collated_callprofiling_types.h"
#include "hwprofiling_types.h"
#ifdef _MPI
#include "mpiprofiling_types.h"
#endif
#ifdef _OMP
#include "ompprofiling_types.h"
#endif
#ifdef _CUDA
#include "collated_cudaprofiling_types.h"
#endif
#ifdef _ACCPROF
#include "collated_accprofiling_types.h"
#endif

typedef struct {
   collated_callprofile_t callprof;
   hwprofile_t hwprof;
#ifdef _MPI
   mpiprofile_t mpiprof;
#endif
#ifdef _OMP
   ompprofile_t ompprof;
#endif
#ifdef _CUDA
   collated_cudaprofile_t cudaprof;
#endif
#ifdef _ACCPROF
   collated_accprofile_t accprof;
#endif
} collated_profile_t;

#endif
