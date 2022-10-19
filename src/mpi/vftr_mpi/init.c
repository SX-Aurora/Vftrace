#include <mpi.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "mpiprofiling.h"

int vftr_MPI_Init(int *argc, char ***argv) {
   SELF_PROFILE_START_FUNCTION;
   int retVal = PMPI_Init(argc, argv);

   PMPI_Comm_size(MPI_COMM_WORLD, &vftrace.process.nprocesses);
   PMPI_Comm_rank(MPI_COMM_WORLD, &vftrace.process.processID);

   vftr_create_profiled_ranks_list(vftrace.environment,
                                   vftrace.process,
                                   &vftrace.mpi_state);
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
