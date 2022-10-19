#ifdef _MPI
#include <mpi.h>

#include "fetch_and_op.h"

void vftr_MPI_Fetch_and_op_f2vftr(const void *origin_addr, void *result_addr,
                                  MPI_Fint *f_datatype, MPI_Fint *target_rank,
                                  MPI_Aint *target_disp, MPI_Fint *f_op,
                                  MPI_Fint *f_win, MPI_Fint *f_error) {

   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Op c_op = PMPI_Op_f2c(*f_op);
   MPI_Win c_win = PMPI_Win_f2c(*f_win);

   int c_error = vftr_MPI_Fetch_and_op(origin_addr,
                                       result_addr,
                                       c_datatype,
                                       (int)(*target_rank),
                                       *target_disp,
                                       c_op,
                                       c_win);

   *f_error = (MPI_Fint) c_error;
}

#endif
