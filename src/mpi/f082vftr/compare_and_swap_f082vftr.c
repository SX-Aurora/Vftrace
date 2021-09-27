/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

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
