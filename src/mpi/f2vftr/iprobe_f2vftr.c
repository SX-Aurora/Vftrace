#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "iprobe.h"

void vftr_MPI_Iprobe_f2vftr(MPI_Fint *source, MPI_Fint *tag, MPI_Fint *f_comm,
                            MPI_Fint *f_flag, MPI_Fint *f_status, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Status c_status;
   int c_flag;

   int c_error = vftr_MPI_Iprobe((int)(*source),
                                 (int)(*tag),
                                 c_comm,
                                 &c_flag,
                                 &c_status);

   *f_flag = (MPI_Fint) c_flag;
   if (f_status != MPI_F_STATUS_IGNORE) {
      PMPI_Status_c2f(&c_status, f_status);
   }

   *f_error = (MPI_Fint) c_error;
}

#endif
