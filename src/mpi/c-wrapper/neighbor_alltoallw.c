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

#include "mpi_logging.h"
#include "vftr_regions.h"
#include "vftr_environment.h"
#include "neighbor_alltoallw_c2vftr.h"

int MPI_Neighbor_alltoallw(const void *sendbuf, const int *sendcounts,
                           const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                           void *recvbuf, const int *recvcounts,
                           const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                           MPI_Comm comm) {
   // Estimate synchronization time
   if (vftr_environment.mpi_show_sync_time->value) {
      vftr_internal_region_begin("MPI_Neighbor_alltoallw_sync");
      PMPI_Barrier(comm);
      vftr_internal_region_end("MPI_Neighbor_alltoallw_sync");
   }
   if (vftr_no_mpi_logging()) {
      return PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                     recvbuf, recvcounts, rdispls, recvtypes,
                                     comm);
   } else {
      return vftr_MPI_Neighbor_alltoallw_c2vftr(sendbuf, sendcounts, sdispls, sendtypes,
                                                recvbuf, recvcounts, rdispls, recvtypes,
                                                comm);
   }
}

#endif
