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
#include "vftr_regions.h"
#include "vftr_environment.h"
#include "vftr_buf_addr_const.h"
#include "alltoall.h"

int vftr_MPI_Alltoall_c2vftr(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             MPI_Comm comm) {
   // Estimate synchronization time
   if (vftr_environment.mpi_show_sync_time->value) {
      vftr_internal_region_begin("MPI_Alltoall_sync");
      PMPI_Barrier(comm);
      vftr_internal_region_end("MPI_Alltoall_sync");
   }
   if (vftr_no_mpi_logging()) {
      return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                           recvcount, recvtype, comm);
   } else {
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         return vftr_MPI_Alltoall_intercom(sendbuf, sendcount, sendtype,
                                           recvbuf, recvcount, recvtype,
                                           comm);
      } else {
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            return vftr_MPI_Alltoall_inplace(sendbuf, sendcount, sendtype,
                                             recvbuf, recvcount, recvtype,
                                             comm);
         } else {
            return vftr_MPI_Alltoall(sendbuf, sendcount, sendtype,
                                     recvbuf, recvcount, recvtype,
                                     comm);
         }

      }
   }
}

#endif
