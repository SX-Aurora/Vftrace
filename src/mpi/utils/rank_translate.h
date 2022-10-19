#ifndef RANK_TRANSLATE_H
#define RANK_TRANSLATE_H

#include <mpi.h>

// Translate a rank from a local group to the global rank
int vftr_local2global_rank(MPI_Comm comm, int local_rank);

// Translate a rank from a remote group to the global rank
int vftr_remote2global_rank(MPI_Comm comm, int remote_rank);

#endif
