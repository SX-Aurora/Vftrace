#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "accumulate_c2vftr.h"

int MPI_Accumulate(const void *origin_addr, int origin_count,
                   MPI_Datatype origin_datatype, int target_rank,
                   MPI_Aint target_disp, int target_count,
                   MPI_Datatype target_datatype, MPI_Op op,
                   MPI_Win win) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Accumulate(origin_addr, origin_count, origin_datatype,
                             target_rank, target_disp, target_count,
                             target_datatype, op, win);
   } else {
      return vftr_MPI_Accumulate_c2vftr(origin_addr, origin_count, origin_datatype,
                                        target_rank, target_disp, target_count,
                                        target_datatype, op, win);
   }
}

#endif
