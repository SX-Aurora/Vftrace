#include <mpi.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "vftr_finalize.h"
#include "mpiprofiling.h"
#include "requests.h"

int vftr_MPI_Finalize() {
   SELF_PROFILE_START_FUNCTION;
   vftr_free_request_list(&vftrace.mpi_state);
   vftr_free_profiled_ranks_list(&vftrace.mpi_state);
   SELF_PROFILE_END_FUNCTION;
   // it is neccessary to finalize vftrace here, in order to properly communicat stack ids
   // between processes. After MPI_Finalize communication between processes is prohibited
   vftr_finalize();

   return PMPI_Finalize();
}
