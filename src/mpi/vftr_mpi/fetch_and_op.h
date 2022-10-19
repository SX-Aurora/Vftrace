#ifndef FETCH_AND_OP_H
#define FETCH_AND_OP_H

#include <mpi.h>

int vftr_MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                          MPI_Datatype datatype, int target_rank,
                          MPI_Aint target_disp, MPI_Op op, MPI_Win win);

#endif
