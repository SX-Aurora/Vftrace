#ifndef PAPIPROFILING_TYPES_H
#define PAPIPROFILING_TYPES_H

typedef struct {
   long long *counters_incl;
   long long *counters_excl;
} papiprofile_t;

#endif
