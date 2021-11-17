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

#include <stdlib.h>

#include "test.h"
  
void vftr_MPI_Test_f082vftr(MPI_Fint *f_request, MPI_Fint *f_flag,
                            MPI_F08_status *f_status, MPI_Fint *f_error) {

   MPI_Request c_request;
   c_request = PMPI_Request_f2c(*f_request);
   int c_flag;
   MPI_Status c_status;

   int c_error = vftr_MPI_Test(&c_request,
                               &c_flag,
                               &c_status);

   *f_request = PMPI_Request_c2f(c_request);
   *f_flag = (MPI_Fint) c_flag;
   if (f_status != MPI_F08_STATUS_IGNORE) {
      PMPI_Status_c2f08(&c_status, f_status);
   }
   *f_error = (MPI_Fint) c_error;
}

#endif
