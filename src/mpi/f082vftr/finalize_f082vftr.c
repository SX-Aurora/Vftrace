#ifdef _MPI
#include <mpi.h>

#include "finalize.h"

void vftr_MPI_Finalize_f082vftr(MPI_Fint *f_error) {
   int c_error = vftr_MPI_Finalize();
   *f_error = (MPI_Fint) (c_error);
}

#endif
