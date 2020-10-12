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

#ifndef VFTR_MPI_ISSEND_F_H
#define VFTR_MPI_ISSEND_F_H

#ifdef _MPI
#include <mpi.h>

void vftr_MPI_Issend_F(void *buf, MPI_Fint *count, MPI_Fint *f_datatype,
                       MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *f_comm,
                       MPI_Fint *f_request, MPI_Fint *f_error);

#endif
#endif
