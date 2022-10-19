#ifdef _MPI
#include <stdlib.h>

#include <mpi.h>

#include "init_thread.h"

void vftr_MPI_Init_thread_f082vftr(MPI_Fint *f_required,
                                   MPI_Fint *f_provided,
                                   MPI_Fint *f_error) {
   int c_required = (int)(*f_required);
   int c_provided;
   int c_error = vftr_MPI_Init_thread(NULL, NULL, c_required, &c_provided);
   *f_provided = (MPI_Fint) (c_provided);
   *f_error = (MPI_Fint) (c_error);
}

#endif
