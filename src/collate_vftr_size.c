#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "vftrace_state.h"

void vftr_collate_vftr_size(vftrace_t *vftrace) {
   vftrace->size.rank_wise = vftr_sizeof_vftrace_t(*vftrace);
#ifdef _MPI
   MPI_Reduce(&(vftrace->size.rank_wise),
              &(vftrace->size.total),
              1,
              MPI_LONG_LONG,
              MPI_SUM,
              0,
              MPI_COMM_WORLD);
#else
   vftrace->size.total = vftrace->size.rank_wise;
#endif
}
