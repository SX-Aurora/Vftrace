#ifdef _MPI
#include <mpi.h>

#include "mpi_logging.h"
#include "compare_and_swap_c2vftr.h"

int MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                         void *result_addr, MPI_Datatype datatype,
                         int target_rank, MPI_Aint target_disp,
                         MPI_Win win) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr,
                                   datatype, target_rank, target_disp, win);
   } else {
      return vftr_MPI_Compare_and_swap_c2vftr(origin_addr, compare_addr, result_addr,
                                              datatype, target_rank, target_disp, win);
   }
}

#endif
