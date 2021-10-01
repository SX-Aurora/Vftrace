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

#include "vftr_environment.h"
#include "vftr_pause.h"
#include "pcontrol.h"

// determine based on several criteria if
// the communication should just be executed or also logged
bool vftr_no_mpi_logging() {
   return vftrace_Pcontrol_level == 0 || 
          vftr_off() ||
          !vftr_environment.mpi_log->value ||
          vftr_paused;
}

// int version of above function for well defined fortran-interoperability
int vftr_no_mpi_logging_int() {
   return vftr_no_mpi_logging() ? 1 : 0;
}
