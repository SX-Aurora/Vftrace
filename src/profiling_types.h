#ifndef PROFILING_TYPES_H
#define PROFILING_TYPES_H

#include "callprofiling_types.h"
#include "overheadprofiling_types.h"
#ifdef _MPI
#include "mpiprofiling_types.h"
#endif

typedef struct {
   int threadID;
   callProfile_t callProf;
#ifdef _MPI
   mpiProfile_t mpiProf;
#endif
   overheadProfile_t overheadProf;
} profile_t;

// each thread gets their own profile during runtime
typedef struct {
   int nprofiles;
   int maxprofiles;
   profile_t* profiles;
} profilelist_t;

#endif
