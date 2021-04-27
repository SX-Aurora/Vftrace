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

#include <vftr_mpi_buf_addr_const.h>
#include <vftr_mpi_bcast.h>

void vftr_MPI_Bcast_f2c(void *buffer, MPI_Fint *count, MPI_Fint *f_datatype, 
                        MPI_Fint *root, MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   int c_error = vftr_MPI_Bcast(buffer,
                                (int)(*count),
                                c_datatype,
                                (int)(*root),
                                c_comm);

   *f_error = (MPI_Fint) (c_error);
}

#endif
