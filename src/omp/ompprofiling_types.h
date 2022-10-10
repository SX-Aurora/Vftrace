#ifndef OMPPROFILING_TYPES_H
#define OMPPROFILING_TYPES_H

typedef struct {
   // overhead from vftrace OMP-book keeping
   long long overhead_nsec;
} ompprofile_t;
#endif
