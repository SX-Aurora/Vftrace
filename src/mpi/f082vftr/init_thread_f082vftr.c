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
