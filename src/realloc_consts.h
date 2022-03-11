#ifndef REALLOC_CONSTS_H
#define REALLOC_CONSTS_H

// constants that determine the reallocation speed
// of dynamic arrays in vftrace
//
// maxn = maxn*vftr_realloc_rate+vftr_realloc_add

extern const float vftr_realloc_rate;

extern const int vftr_realloc_add;

#endif
