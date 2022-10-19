#ifdef _MPI
#include <mpi.h>

#include "bsend.h"

void vftr_MPI_Bsend_f2vftr(void *buf, MPI_Fint *count, MPI_Fint *f_datatype,
                           MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *f_comm,
                           MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);

   int c_error = vftr_MPI_Bsend(buf,
                                (int)(*count),
                                c_datatype,
                                (int)(*dest),
                                (int)(*tag),
                                c_comm);

   *f_error = (MPI_Fint) (c_error);
}

#endif
