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

#ifndef VFTR_MPI_UTIL_H
#define VFTR_MPI_UTIL_H

#include <stdbool.h>

#ifdef _MPI
#include <mpi.h>
#endif

typedef enum vftr_direction_t {
   send,
   recv
} vftr_direction;

#ifdef _MPI
// Adjust vftrace to the just initialized MPI-environment
void vftr_after_mpi_init();

// Translates an MPI-Datatype into the vftr type index
int vftr_get_mpitype_idx(MPI_Datatype mpi_type);

// Converts an mpi-datatype into a name string for that type
const char *vftr_get_mpitype_string(MPI_Datatype mpi_type);

// Translate a rank from a local group to the global rank
int vftr_local2global_rank(MPI_Comm comm, int local_rank);

// Translate a rank from a remote group
int vftr_remote2global_rank(MPI_Comm comm, int remote_rank);

#endif

// Converts an mpi-datatype into a name string for that type
const char *vftr_get_mpitype_string_from_idx(int mpi_type_idx);

#ifdef _MPI
// mark a MPI_Status as empty
void vftr_empty_mpi_status(MPI_Status *status);

bool vftr_mpi_status_is_empty(MPI_Status *status);

// check if a request is active
bool vftr_mpi_request_is_active(MPI_Request request);

#endif

#endif
