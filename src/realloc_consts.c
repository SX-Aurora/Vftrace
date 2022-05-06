// constants that determine the reallocation speed
// of dynamic arrays in vftrace
//
// maxn = maxn*vftr_realloc_rate+vftr_realloc_add
//
// vftr_realloc_add needs to be > 0,
// because if maxn is [0,2]
// it will not be changed by the rate

const float vftr_realloc_rate = 1.4142;
const int vftr_realloc_add = 2;
