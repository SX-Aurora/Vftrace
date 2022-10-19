#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "fetch_and_op_c2vftr.h"

int MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                     MPI_Datatype datatype, int target_rank,
                     MPI_Aint target_disp, MPI_Op op, MPI_Win win) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Fetch_and_op(origin_addr, result_addr, datatype,
                               target_rank, target_disp, op, win);
   } else {
      return vftr_MPI_Fetch_and_op_c2vftr(origin_addr, result_addr, datatype,
                                          target_rank, target_disp, op, win);
   }
}

#endif
