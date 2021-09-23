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

#include <stdbool.h>

#ifndef VFTR_MPI_BUF_ADDR_CONST_H
#define VFTR_MPI_BUF_ADDR_CONST_H

#ifdef _MPI
#include <mpi.h>

// check if the given address is the special MPI_BOTTOM handle
bool vftr_is_C_MPI_BOTTOM(const void *addr);

bool vftr_is_F_MPI_BOTTOM(const void *addr);

// check if the given address is the special MPI_IN_PLACE handle
bool vftr_is_C_MPI_IN_PLACE(const void *addr);

bool vftr_is_F_MPI_IN_PLACE(const void *addr);

#endif
#endif
