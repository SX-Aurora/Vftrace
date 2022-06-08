#ifndef OVERHEADPROFILING_TYPES_H
#define OVERHEADPROFILING_TYPES_H

typedef struct {
   // accumulated overhead time
   long long hook_usec;
#ifdef _MPI
   long long mpi_usec;
#endif
#ifdef _OMP
   long long omp_usec;
#endif
} overheadProfile_t;

#endif
