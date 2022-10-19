#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "probe.h"

void vftr_MPI_Probe_f082vftr(MPI_Fint *source, MPI_Fint *tag, MPI_Fint *f_comm,
                             MPI_F08_status *f_status, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Status c_status;

   int c_error = vftr_MPI_Probe((int)(*source),
                                (int)(*tag),
                                c_comm,
                                &c_status);

   if (f_status != MPI_F08_STATUS_IGNORE) {
      PMPI_Status_c2f08(&c_status, f_status);
   }

   *f_error = (MPI_Fint) c_error;
}

#endif
