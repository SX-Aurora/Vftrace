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

#include "vftr_mpi_environment.h"
#include "vftr_mpi_utils.h"
#include "vftr_setup.h"

int MPI_Init(int *argc, char ***argv) {

   int returnValue = PMPI_Init(argc, argv);

   vftr_after_mpi_init();

   return returnValue;
}

int MPI_Finalize() {

   // it is neccessary to finalize vftrace here, in order to properly communicat stack ids
   // between processes. After MPI_Finalize communication between processes is prohibited
   vftr_finalize();

   return PMPI_Finalize();
}

int MPI_Pcontrol(const int level, ...) {
   return vftr_MPI_Pcontrol(level);
}

#endif
