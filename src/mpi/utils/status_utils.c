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

#include <mpi.h>

// mark a MPI_Status as empty
void vftr_empty_mpi_status(MPI_Status *status) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // A status is empty if its members have the following values:
   // MPI_TAG == MPI_ANY_TAG, MPI_SOURCE == MPI_ANY_SOURCE, MPI_ERROR == MPI_SUCCESS
   status->MPI_TAG = MPI_ANY_TAG;
   status->MPI_SOURCE = MPI_ANY_SOURCE;
   status->MPI_ERROR = MPI_SUCCESS;
   return;
}

// check if a status is empty
bool vftr_mpi_status_is_empty(MPI_Status *status) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // a status is empty if its members have the following values:
   // MPI_TAG == MPI_ANY_TAG, MPI_SOURCE == MPI_ANY_SOURCE, MPI_ERROR == MPI_SUCCESS
   return (status->MPI_TAG == MPI_ANY_TAG &&
           status->MPI_SOURCE == MPI_ANY_SOURCE &&
           status->MPI_ERROR == MPI_SUCCESS);
}
