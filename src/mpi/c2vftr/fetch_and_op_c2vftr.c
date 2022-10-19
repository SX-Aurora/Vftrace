#ifdef _MPI
#include <mpi.h>

#include "fetch_and_op.h"

int vftr_MPI_Fetch_and_op_c2vftr(const void *origin_addr, void *result_addr,
                                 MPI_Datatype datatype, int target_rank,
                                 MPI_Aint target_disp, MPI_Op op, MPI_Win win) {
   return vftr_MPI_Fetch_and_op(origin_addr, result_addr, datatype,
                                target_rank, target_disp, op, win);
}

#endif
