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

#include "vftrace_state.h"
#include "sampling_types.h"
#include "mpi_util_types.h"

// determine based on several criteria if
// the communication should just be executed or also logged
bool vftr_no_mpi_logging() {
   return vftrace.mpi_state.pcontrol_level == 0 ||
          vftrace.state == off ||
          !vftrace.environment.mpi_log.value.bool_val ||
          vftrace.state == paused;
}

// int version of above function for well defined fortran-interoperability
int vftr_no_mpi_logging_int() {
   return vftr_no_mpi_logging() ? 1 : 0;
}

// Store the message information in a vfd file
void vftr_store_message_info(message_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend,
                             int stackID) {

   FILE *fp = vftrace.sampling.vfdfilefp;
   sample_kind kind = samp_message;
   fwrite(&kind, sizeof(sample_kind), 1, fp);
   fwrite(&dir, sizeof(message_direction), 1, fp);
   fwrite(&rank, sizeof(int), 1, fp);
   fwrite(&type_idx, sizeof(int), 1, fp);
   fwrite(&count, sizeof(int), 1, fp);
   fwrite(&type_size, sizeof(int), 1, fp);
   fwrite(&tag, sizeof(int), 1, fp);
   fwrite(&tstart, sizeof(long long), 1, fp);
   fwrite(&tend, sizeof(long long), 1, fp);
   fwrite(&stackID, sizeof(int), 1, fp);

   vftrace.sampling.message_samplecount++;
}
