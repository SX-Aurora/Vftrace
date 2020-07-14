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
// PControl level as required by the MPI-Standard for profiling interfaces
int vftrace_Pcontrol_level = 1;

// vftrace internal routine to control the profiling level
int vftr_MPI_Pcontrol(const int level) {
   // level == 0 Profiling is disabled
   // level == 1 Profiling is enabled at a normal default level of detail
   // lebel == 2 Buffers are flushed, which may be a no-op in some profilers
   // All other values have profile library defined effects and additional arguments
   if ((level >=0) && (level <=2)) {
      vftrace_Pcontrol_level = level;
      return 0;
   }

   return 1;
}
#endif
