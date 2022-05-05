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

#ifndef MPI_LOGGING_H
#define MPI_LOGGING_H

#include <stdbool.h>

#include "mpi_util_types.h"

// determine based on several criteria if
// the communication should just be executed or also logged
bool vftr_no_mpi_logging();

// int version of above function for well defined fortran-interoperability
int vftr_no_mpi_logging_int();

// Store the message information in a vfd file
void vftr_store_message_info(message_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend,
                             int stackID);

#endif
