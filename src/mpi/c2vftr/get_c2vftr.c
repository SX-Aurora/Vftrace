#ifdef _MPI
#include <mpi.h>

#include "get.h"

int vftr_MPI_Get_c2vftr(void *origin_addr, int origin_count,
                        MPI_Datatype origin_datatype, int target_rank,
                        MPI_Aint target_disp, int target_count,
                        MPI_Datatype target_datatype, MPI_Win win) {
   return vftr_MPI_Get(origin_addr, origin_count, origin_datatype,
                       target_rank, target_disp, target_count,
                       target_datatype, win);
}

#endif
