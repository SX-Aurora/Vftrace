#ifdef _MPI
#include <mpi.h>

#include "compare_and_swap.h"

int vftr_MPI_Compare_and_swap_c2vftr(const void *origin_addr, const void *compare_addr,
                                     void *result_addr, MPI_Datatype datatype,
                                     int target_rank, MPI_Aint target_disp,
                                     MPI_Win win) {
   return vftr_MPI_Compare_and_swap(origin_addr, compare_addr, result_addr,
                                    datatype, target_rank, target_disp, win);
}

#endif
