#ifndef COMPARE_AND_SWAP_C2VFTR_H
#define COMPARE_AND_SWAP_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Compare_and_swap_c2vftr(const void *origin_addr, const void *compare_addr,
                                     void *result_addr, MPI_Datatype datatype,
                                     int target_rank, MPI_Aint target_disp,
                                     MPI_Win win);

#endif
#endif
