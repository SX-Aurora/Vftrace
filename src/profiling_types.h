#ifndef PROFILING_TYPES_H
#define PROFILING_TYPES_H

#include "callprofiling_types.h"
#ifdef _MPI
#include "mpiprofiling_types.h"
#endif

typedef struct {
   int threadID;
   callprofile_t callprof;
#ifdef _MPI
   mpiprofile_t mpiprof;
#endif
} profile_t;

// each thread gets their own profile during runtime
typedef struct {
   int nprofiles;
   int maxprofiles;
   profile_t* profiles;
} profilelist_t;

#endif
