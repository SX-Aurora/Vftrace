#ifndef CALLPROFILING_TYPES_H
#define CALLPROFILING_TYPES_H

typedef struct {
   // number of calls
   long long calls;
   // time spend in the function (including subfunctions)
   long long time_nsec;
   // time spend in the function (excluding subfunctions)
   // computed during final stack update
   long long time_excl_nsec;
   // calloverhead induced by vftrace stack bookkeeping
   long long overhead_nsec;
} callProfile_t;

#endif
