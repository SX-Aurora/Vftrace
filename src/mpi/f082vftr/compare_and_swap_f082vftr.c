#ifdef _MPI
#include <mpi.h>

#include "compare_and_swap.h"

void vftr_MPI_Compare_and_swap_f082vftr(const void *origin_addr,
                                        const void *compare_addr,
                                        void *result_addr, MPI_Fint *f_datatype,
                                        MPI_Fint *target_rank, MPI_Aint *target_disp,
                                        MPI_Fint *f_win, MPI_Fint *f_error) {

   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Win c_win = PMPI_Win_f2c(*f_win);

   int c_error = vftr_MPI_Compare_and_swap(origin_addr,
                                           compare_addr,
                                           result_addr,
                                           c_datatype,
                                           (int)(*target_rank),
                                           *target_disp,
                                           c_win);

   *f_error = (MPI_Fint) c_error;
}

#endif
