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

#include "vftrace_state.h"

#include "self_profile.h"
// vftrace internal routine to control the profiling level
int vftr_MPI_Pcontrol(const int level) {
   SELF_PROFILE_START_FUNCTION;
   // level == 0 Profiling is disabled
   // level == 1 Profiling is enabled at a normal default level of detail
   // lebel == 2 Buffers are flushed, which may be a no-op in some profilers
   // All other values have profile library defined effects and additional arguments
   int retVal = 1;
   if ((level >=0) && (level <=2)) {
      vftrace.mpi_state.pcontrol_level = level;
      retVal = 0;
   }

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
