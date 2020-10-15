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

#include "vftr_mpi_put.h"

void vftr_MPI_Put_F(void *origin_addr, MPI_Fint *origin_count,
                    MPI_Fint *f_origin_datatype, MPI_Fint *target_rank,
                    MPI_Aint *target_disp, MPI_Fint *target_count,
                    MPI_Fint *f_target_datatype, MPI_Fint *f_win,
                    MPI_Fint *f_error) {

   MPI_Datatype c_origin_datatype = PMPI_Type_f2c(*f_origin_datatype);
   MPI_Datatype c_target_datatype = PMPI_Type_f2c(*f_target_datatype);
   MPI_Win c_win = PMPI_Win_f2c(*f_win);

   int c_error = vftr_MPI_Put(origin_addr,
                              (int)(*origin_count),
                              c_origin_datatype,
                              (int)(*target_rank),
                              *target_disp,
                              (int)(*target_count),
                              c_target_datatype,
                              c_win);

   *f_error = (MPI_Fint) c_error;
}

#endif
