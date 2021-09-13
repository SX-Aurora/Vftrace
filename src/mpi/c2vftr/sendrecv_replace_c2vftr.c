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

#include "vftr_mpi_utils.h"
#include "sendrecv_replace.h"

int vftr_MPI_Sendrecv_replace_c2vftr(void *buf, int count, MPI_Datatype datatype,
                                     int dest, int sendtag, int source, int recvtag,
                                     MPI_Comm comm, MPI_Status *status) {
   if (vftr_no_mpi_logging()) {
      return PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag,
                                   source, recvtag, comm, status);
   } else {
      return vftr_MPI_Sendrecv_replace(buf, count, datatype, dest, sendtag,
                                       source, recvtag, comm, status);
   }
}

#endif