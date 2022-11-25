#ifndef VEDAPROFILING_TYPES
#define VEDAPROFILING_TYPES

typedef struct {
   int n_calls;
   long long HtoD_bytes;
   long long DtoH_bytes;
   long long H_bytes;
   double acc_HtoD_bw;
   double acc_DtoH_bw;
   double acc_H_bw;
   long long total_time_nsec;
   long long overhead_nsec;
} vedaprofile_t;

#endif
