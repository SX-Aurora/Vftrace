#ifndef VEDAPROFILING_TYPES
#define VEDAPROFILING_TYPES

typedef struct {
   int n_calls;
   size_t HtoD_bytes;
   size_t DtoH_bytes;
   size_t H_bytes;
   double acc_HtoD_bw;
   double acc_DtoH_bw;
   double acc_H_bw;
   long long total_time_nsec;
   size_t overhead_nsec;
} vedaprofile_t;

#endif
