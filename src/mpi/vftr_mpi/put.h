#ifndef PUT_H
#define PUT_H

#include <mpi.h>

int vftr_MPI_Put(const void *origin_addr, int origin_count,
                 MPI_Datatype origin_datatype, int target_rank,
                 MPI_Aint target_disp, int target_count,
                 MPI_Datatype target_datatype, MPI_Win win);

#endif
