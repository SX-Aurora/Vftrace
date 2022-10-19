#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "init.h"

void vftr_MPI_Init_f2vftr(MPI_Fint *f_error) {
   int c_error = vftr_MPI_Init(NULL, NULL);
   *f_error = (MPI_Fint) (c_error);
}

#endif
