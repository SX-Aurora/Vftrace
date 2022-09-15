#ifndef COLLATED_CALLPROFILING_TYPES_H
#define COLLATED_CALLPROFILING_TYPES_H

typedef struct {
   // number of calls
   long long calls;
   // time spend in the function (including subfunctions)
   long long time_usec;
   // time spend in the function (excluding subfunctions)
   // computed during final stack update
   long long time_excl_usec;
   // calloverhead induced by vftrace stack bookkeeping
   long long overhead_usec;
   // additional info to compute the imbalances across ranks
   int on_nranks;
   int max_on_rank;
   int min_on_rank;
   long long average_time_usec;
   long long max_time_usec;
   long long min_time_usec;
   double max_imbalance;
   int max_imbalance_on_rank;
} collated_callProfile_t;

#endif
