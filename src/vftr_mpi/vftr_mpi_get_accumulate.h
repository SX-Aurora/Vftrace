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

#ifndef VFTR_MPI_GET_H
#define VFTR_MPI_GET_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Get_accumulate(const void *origin_addr, int origin_count,
                            MPI_Datatype origin_datatype, void *result_addr,
                            int result_count, MPI_Datatype result_datatype,
                            int target_rank, MPI_Aint target_disp, int target_count,
                            MPI_Datatype target_datatype, MPI_Op op, MPI_Win win);

#endif
#endif
