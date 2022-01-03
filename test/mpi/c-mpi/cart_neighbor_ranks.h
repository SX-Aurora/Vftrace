#ifndef CART_NEIGHBOR_RANKS_H
#define CART_NEIGHBOR_RANKS_H

#include <mpi.h>

void cart_neighbor_ranks(MPI_Comm cart_comm,
                         int *nneighbors_ptr,
                         int **neighbors_ptr);

#endif
