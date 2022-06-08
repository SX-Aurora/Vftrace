#ifndef CALLPROFILING_TYPES_H
#define CALLPROFILING_TYPES_H

typedef struct {
   // number of calls
   long long calls;
   // cycles spend in the function (including subfunctions)
   long long cycles;
   // time spend in the function (including subfunctions)
   long long time_usec;
   // time spend in the function (excluding subfunctions)
   // computed during final stack update
   long long time_excl_usec;
} callProfile_t;

#endif
