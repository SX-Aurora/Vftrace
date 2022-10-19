#ifndef RPUT_C2VFTR_H
#define RPUT_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Rput_c2vftr(const void *origin_addr, int origin_count,
                         MPI_Datatype origin_datatype, int target_rank,
                         MPI_Aint target_disp, int target_count,
                         MPI_Datatype target_datatype, MPI_Win win,
                         MPI_Request *request);

#endif
#endif
