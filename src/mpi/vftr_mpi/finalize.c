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

#include <mpi.h>

#include "vftr_setup.h"
#include "requests.h"

int vftr_MPI_Finalize() {

   vftr_free_request_list();
   // it is neccessary to finalize vftrace here, in order to properly communicat stack ids
   // between processes. After MPI_Finalize communication between processes is prohibited
   vftr_finalize();

   return PMPI_Finalize();
}
