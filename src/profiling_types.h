#ifndef PROFILING_TYPES_H
#define PROFILING_TYPES_H

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
   // cycles spend in the function (including subfunctions)
   long long cycles;
   // time spend in the function (including subfunctions)
   long long time_usec;
} callProfile_t;

typedef struct {
   callProfile_t callProf;
   memoryProfile_t memProf;
} profile_t;

#endif
