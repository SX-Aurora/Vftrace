#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "barrier.h"

void vftr_MPI_Barrier_f082vftr(MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   int c_error = vftr_MPI_Barrier(c_comm);

   *f_error = (MPI_Fint) c_error;
}

#endif
