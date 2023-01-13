#ifndef HWPROFILING_TYPES_H
#define HWPROFILING_TYPES_H

typedef struct {
   long long *counters_incl;
   long long *counters_excl;
   double *observables;
} hwprofile_t;

#endif
