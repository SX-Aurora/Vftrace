#include "vftrace_state.h"

#include "self_profile.h"
// vftrace internal routine to control the profiling level
int vftr_MPI_Pcontrol(const int level) {
   SELF_PROFILE_START_FUNCTION;
   // level == 0 profiling is disabled
   // level == 1 profiling is enabled at a normal default level of detail
   // lebel == 2 Buffers are flushed, which may be a no-op in some profilers
   // All other values have profile library defined effects and additional arguments
   int retVal = 1;
   if ((level >=0) && (level <=2)) {
      vftrace.mpi_state.pcontrol_level = level;
      retVal = 0;
   }

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
