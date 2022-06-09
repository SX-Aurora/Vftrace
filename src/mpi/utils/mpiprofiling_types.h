#ifndef MPIPROFILING_TYPES_H
#define MPIPROFILING_TYPES_H

typedef struct {
   // amount of bytes send/recv by this stack
   long long send_bytes;
   long long recv_bytes;
} mpiProfile_t;
#endif
