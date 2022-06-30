#include "mpi_util_types.h"
#include "mpiprofiling_types.h"

mpiProfile_t vftr_new_mpiprofiling() {
   mpiProfile_t prof;
   prof.nmessages = 0ll;
   prof.send_bytes = 0ll;
   prof.recv_bytes = 0ll;
   prof.acc_send_bw = 0.0;
   prof.acc_recv_bw = 0.0;
   prof.total_time_usec = 0ll;

   return prof;
}

void vftr_accumulate_message_info(mpiProfile_t *prof_ptr,
                                  message_direction dir, long long count,
                                  int type_idx, int type_size, int rank,
                                  int tag, long long tstart,
                                  long long tend) {
   (void) type_idx;
   (void) rank;
   (void) tag;
   (void) tstart;
   (void) tend;

   int nbytes = count * type_size;
   long long time = tend - tstart;
   double bw = nbytes * 1.0e-6 / time;

   prof_ptr->nmessages++;
   if (dir == send) {
      prof_ptr->send_bytes += nbytes;
      prof_ptr->acc_send_bw += bw;
   } else {
      prof_ptr->recv_bytes += nbytes;
      prof_ptr->acc_recv_bw += bw;
   }
   prof_ptr->total_time_usec += time;
}

void vftr_mpiprofiling_free(mpiProfile_t *prof_ptr) {
   (void) prof_ptr;
}
