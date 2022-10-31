#include <stdbool.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "sampling_types.h"
#include "mpi_util_types.h"

// determine based on several criteria if
// the communication should just be executed or also logged
bool vftr_no_mpi_logging() {
   return vftrace.mpi_state.pcontrol_level == 0 ||
          vftrace.state == off ||
          !vftrace.config.mpi.log_messages.value ||
          vftrace.state == paused;
}

// int version of above function for well defined fortran-interoperability
int vftr_no_mpi_logging_int() {
   return vftr_no_mpi_logging() ? 1 : 0;
}

// write the message information to the vfd-file
void vftr_write_message_info(message_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend,
                             int stackID, int threadID) {
   SELF_PROFILE_START_FUNCTION;
   FILE *fp = vftrace.sampling.vfdfilefp;
   sample_kind kind = samp_message;
   fwrite(&kind, sizeof(sample_kind), 1, fp);
   fwrite(&dir, sizeof(message_direction), 1, fp);
   fwrite(&rank, sizeof(int), 1, fp);
   fwrite(&type_idx, sizeof(int), 1, fp);
   fwrite(&count, sizeof(int), 1, fp);
   fwrite(&type_size, sizeof(int), 1, fp);
   fwrite(&tag, sizeof(int), 1, fp);
   fwrite(&tstart, sizeof(long long), 1, fp);
   fwrite(&tend, sizeof(long long), 1, fp);
   fwrite(&stackID, sizeof(int), 1, fp);
   fwrite(&threadID, sizeof(int), 1, fp);

   vftrace.sampling.message_samplecount++;
   SELF_PROFILE_END_FUNCTION;
}
