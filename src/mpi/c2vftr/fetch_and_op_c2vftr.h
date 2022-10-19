#ifndef FETCH_AND_OP_C2VFTR_H
#define FETCH_AND_OP_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Fetch_and_op_c2vftr(const void *origin_addr, void *result_addr,
                                 MPI_Datatype datatype, int target_rank,
                                 MPI_Aint target_disp, MPI_Op op, MPI_Win win);

#endif
#endif
