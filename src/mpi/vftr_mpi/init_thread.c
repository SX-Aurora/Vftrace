#include <mpi.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "mpiprofiling.h"

int vftr_MPI_Init_thread(int *argc, char ***argv,
                         int required, int *provided) {
   SELF_PROFILE_START_FUNCTION;
   int retVal = PMPI_Init_thread(argc, argv, required, provided);

   if (!vftrace.config.off.value) {
      PMPI_Comm_size(MPI_COMM_WORLD, &vftrace.process.nprocesses);
      PMPI_Comm_rank(MPI_COMM_WORLD, &vftrace.process.processID);

      vftr_create_profiled_ranks_list(vftrace.config,
                                      vftrace.process,
                                      &vftrace.mpi_state);
   }
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
