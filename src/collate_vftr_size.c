#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "self_profile.h"
#include "vftrace_state.h"

void vftr_collate_vftr_size(vftrace_t *vftrace) {
   SELF_PROFILE_START_FUNCTION;
   vftrace->size.rank_wise = vftr_sizeof_vftrace_t(*vftrace);
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      PMPI_Reduce(&(vftrace->size.rank_wise),
                  &(vftrace->size.total),
                  1,
                  MPI_LONG_LONG,
                  MPI_SUM,
                  0,
                  MPI_COMM_WORLD);
   } else {
      vftrace->size.total = vftrace->size.rank_wise;
   }
#else
   vftrace->size.total = vftrace->size.rank_wise;
#endif
   SELF_PROFILE_END_FUNCTION;
}
