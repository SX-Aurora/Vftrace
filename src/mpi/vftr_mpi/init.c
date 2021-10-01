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

#include "vftr_environment.h"
#include "vftr_functions.h"
#include "vftr_stacks.h"
#include "vftr_setup.h"
#include "vftr_hwcounters.h"
#include "vftr_filewrite.h"

int vftr_MPI_Init(int *argc, char ***argv) {

   int returnValue = PMPI_Init(argc, argv);

   if (!vftr_off()) {
      vftr_reset_counts(vftr_froots);
      if (vftr_env_do_sampling()) {
         vftr_prevsampletime = 0;
         vftr_nextsampletime = 0ll;
         vftr_function_samplecount = 0;
         vftr_message_samplecount = 0;
         vftr_prog_cycles = 0ll;

         fseek (vftr_vfd_file, vftr_samples_offset, SEEK_SET);
      }
   }

   return returnValue;
}
