#ifndef MPIPROFILING_TYPES_H
#define MPIPROFILING_TYPES_H

typedef struct {
   // number of messages
   long long nsendmessages;
   long long nrecvmessages;
   // amount of bytes send/recv by this stack
   long long send_bytes;
   long long recv_bytes;
   // accumulated bandwidth
   double acc_send_bw;
   double acc_recv_bw;
   // accumulated communication time
   long long total_time_usec;
   // overhead from vftrace MPI-book keeping
   long long overhead_usec;
} mpiProfile_t;
#endif
