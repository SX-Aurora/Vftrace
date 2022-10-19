#ifndef CART_COMMS_H
#define CART_COMMS_H

#include <mpi.h>

// get the list of neighbor ranks in a cartesian communicator
void vftr_mpi_cart_neighbor_ranks(MPI_Comm cart_comm,
                                  int *nneighbors_ptr,
                                  int **neighbors_ptr);

#endif
