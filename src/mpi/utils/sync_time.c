#include <mpi.h>

#include "vftrace_state.h"
#include "internal_regions.h"

void vftr_estimate_sync_time(char *routine_name, MPI_Comm comm) {
   if (vftrace.config.mpi.show_sync_time.value) {
      vftr_internal_region_begin(routine_name);
      PMPI_Barrier(comm);
      vftr_internal_region_end(routine_name);
   }
}
