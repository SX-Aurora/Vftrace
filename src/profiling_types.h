#ifndef PROFILING_H
#define PROFILING_H

typedef struct {
   long long entry;
   long long exit;
   long long max;
   long long next_entry;
   long long next_exit;
   int tolerance;
   int increment;
} memoryProfile_t;

typedef struct {
   // number of calls
   long long calls;
   // cycles spend in the function (excluding subfunctions)
   long long cycles;
   // time spend in the function (excluding subfunctions)
   long long time_excl_usec;
   // time spend in the function (including subfunctions)
   long long time_incl_usec;
} callProfile_t;

typedef struct {
   callProfile_t callProf;
   memoryProfile_t memProf;
} profile_t;

#endif
