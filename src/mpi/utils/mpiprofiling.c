#include "mpi_util_types.h"
#include "mpiprofiling_types.h"

mpiProfile_t vftr_new_mpiprofiling() {
   mpiProfile_t prof;
   prof.send_bytes = 0ll;
   prof.recv_bytes = 0ll;
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

   if (dir == send) {
      prof_ptr->send_bytes += count * type_size;
   } else {
      prof_ptr->recv_bytes += count * type_size;
   }
}

void vftr_mpiprofiling_free(mpiProfile_t *prof_ptr) {
   (void) prof_ptr;
}
